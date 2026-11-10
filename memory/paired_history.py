"""Paired accepted RGB / inferred Godot memory; never modifies world state.

capture(db, output, initial_rgb=None) returns a record with history_path.
select(db, camera, limit=1) returns ranked records with distance and angle.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

import av
import numpy as np
from PIL import Image

from whitebox.world import scene_from_db


def catalog(db):
    return deepcopy(db.get_meta('paired_history', []))


def _snapshot(db):
    return dict(revision=db.get_meta('whitebox_revision'), tick=db.get_meta('tick'),
                scene=scene_from_db(db), history=db.get_meta('whitebox_history', []),
                video=db.get_meta('whitebox_last_video'))


def _render(scene_path, output, size):
    renderer = Path(__file__).resolve().parents[1]/'whitebox/godot/render.py'
    subprocess.run([sys.executable, str(renderer), '--scene', str(scene_path),
                    '--output', str(output), '--width', str(size[0]), '--height', str(size[1])],
                   check=True, capture_output=True, text=True, timeout=330)
    return output/'camera.png'


def capture(db, output, initial_rgb=None):
    """Register only successful pairs; retries at the same revision are idempotent.

    last_frame must be the lossless decoded final video frame (PNG), not a
    representative earlier sample. Scene is the committed current DB endpoint.
    Failed attempts can leave diagnostic files but never catalog entries.
    """
    state = _snapshot(db)
    history = state['history']
    source = 'INITIAL_RGB' if not history else 'ACCEPTED_VIDEO_ENDPOINT'
    accepted = None
    if history:
        accepted = history[-1]
        if accepted.get('provenance') != 'AGENT_ACCEPTED_VIDEO_INTERPRETATION':
            raise ValueError('Latest history is not an accepted video commit')
        if accepted.get('revision_after') != state['revision']:
            raise ValueError('Stale accepted revision_after')
        artifacts = accepted.get('artifacts', {})
        if not artifacts.get('video') or artifacts['video'] != state['video']:
            raise ValueError('Video is not the current committed video')
        if accepted.get('end') != state['tick']:
            raise ValueError('Accepted endpoint tick differs from current state')
        if accepted.get('cameras') and accepted['cameras'][-1] != state['scene']['camera']:
            raise ValueError('Accepted endpoint camera differs from current scene')
        if 'after' in accepted and accepted['after'] != state['scene']:
            raise ValueError('Accepted after scene differs from current state')
    for record in catalog(db):
        if record['revision'] == state['revision'] and record['source'] == source:
            if record['tick'] != state['tick'] or record['camera'] != state['scene']['camera']:
                raise ValueError('Existing pair has a different endpoint')
            saved_scene = json.loads(Path(record['paths']['scene.json']).read_text(encoding='utf-8'))
            if saved_scene != state['scene']:
                raise ValueError('Existing pair has a different scene at this revision')
            if accepted is not None and record['verification']['video'] != str(Path(artifacts['video']).resolve()):
                raise ValueError('Existing pair belongs to another accepted video')
            return record
    verification = {}
    if accepted is not None:
        last = None
        count = 0
        with av.open(str(artifacts['video'])) as container:
            if len(container.streams.video) != 1:
                raise ValueError('Expected one accepted video stream')
            for frame in container.decode(video=0):
                if frame.is_corrupt:
                    raise ValueError('Corrupt accepted video frame')
                last = frame.to_image().convert('RGB')
                count += 1
        if last is None:
            raise ValueError('Accepted video is empty')
        with Image.open(artifacts['last_frame']) as reference:
            reference = reference.convert('RGB')
            if reference.size != last.size or not np.array_equal(np.asarray(reference), np.asarray(last)):
                raise ValueError('last_frame does not match decoded accepted video endpoint')
        rgb = last
        verification = dict(video=str(Path(artifacts['video']).resolve()),
                            last_frame=str(Path(artifacts['last_frame']).resolve()),
                            decoded_frames=count, endpoint_pixels_equal=True)
    else:
        if initial_rgb is None:
            raise ValueError('initial_rgb required when no accepted history exists')
        with Image.open(initial_rgb) as image:
            rgb = image.convert('RGB')
        verification = dict(initial_rgb=str(Path(initial_rgb).resolve()))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A fresh attempt directory permits retry after a failed Godot render.
    staging = output/('.capture-'+uuid.uuid4().hex)
    staging.mkdir()
    scene_path = staging/'scene.json'
    scene_path.write_text(json.dumps(state['scene'], indent=2), encoding='utf-8')
    rgb.save(staging/'rgb.png')
    proxy = _render(scene_path, staging/'proxy', rgb.size)
    with Image.open(proxy) as image:
        whitebox = image.convert('RGB')
        if whitebox.size != rgb.size:
            raise ValueError('Godot camera render dimensions differ from RGB')
        whitebox.save(staging/'whitebox.png')
    paired = Image.new('RGB', (rgb.width*2, rgb.height))
    paired.paste(rgb, (0, 0))
    paired.paste(whitebox, (rgb.width, 0))
    paired.save(staging/'history.png')
    record = dict(id=('initial-' if accepted is None else 'accepted-')+uuid.uuid4().hex,
                  revision=state['revision'], tick=state['tick'], camera=deepcopy(state['scene']['camera']),
                  source=source, provenance='INFERRED',
                  provenance_detail='RGB source paired with inferred coarse geometry, not ground truth',
                  verification=verification, history_path=str(output/'history.png'),
                  paths={name: str(output/name) for name in ('scene.json', 'rgb.png', 'whitebox.png', 'history.png', 'record.json')})
    with db.transaction():
        if not db.connection.in_transaction:
            db.connection.execute('BEGIN IMMEDIATE')
        if _snapshot(db) != state:
            raise ValueError('World changed during paired capture; retry at current revision')
        records = catalog(db)
        for prior in records:
            if prior['revision'] == state['revision'] and prior['source'] == source:
                return prior
        if any(Path(p).parent == output for prior in records for p in prior.get('paths', {}).values()):
            raise ValueError('Output directory already belongs to another paired capture')
        if (output/'proxy').exists():
            raise ValueError('Output proxy already exists; choose a fresh output directory')
        (staging/'proxy').replace(output/'proxy')
        for name in ('scene.json', 'rgb.png', 'whitebox.png', 'history.png'):
            (staging/name).replace(output/name)
        (output/'record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        db.set_meta('paired_history', records+[record])
    staging.rmdir()
    return record


def select(db, camera, limit=1):
    """Rank by distance/(1+distance) + angle/180; angle in degrees, no cutoff.

    Distance is in scene units; equally weighted bounded spatial and directional
    costs are reported separately. Most recent revision breaks equal scores.
    """
    if not isinstance(limit, int) or limit < 0:
        raise ValueError('limit must be a nonnegative integer')

    def view(value):
        position = np.asarray(value['position'], float)
        direction = np.asarray(value['target'], float)-position
        if position.shape != (3,) or direction.shape != (3,) or not np.isfinite([position, direction]).all() or np.linalg.norm(direction) == 0:
            raise ValueError('Camera needs finite position and nonzero viewing direction')
        return position, direction/np.linalg.norm(direction)

    position, direction = view(camera)
    candidates = []
    for record in catalog(db):
        p, d = view(record['camera'])
        distance = float(np.linalg.norm(p-position))
        angle = math.degrees(math.acos(float(np.clip(d@direction, -1, 1))))
        terms = dict(position=distance/(1+distance), direction=angle/180)
        candidates.append(dict(record, distance=distance, angle=angle, score=sum(terms.values()),
                               ranking=terms))
    return sorted(candidates, key=lambda r: (r['score'], -r['revision'], r['id']))[:limit]
