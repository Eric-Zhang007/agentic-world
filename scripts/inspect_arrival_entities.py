"""Real cached-model boundary evidence, with no world mutation or inferred hit events."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.detector import PerceptionPipeline
from perception.frame_sampler import sample_frames
from scripts.evaluate_motion_ablation import assign, expected_actors


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(a.config.read_text())
    pipeline = PerceptionPipeline({**cfg['perception'], 'device': a.device,
                                  'categories': ['monkey warrior', 'tiger warrior', 'staff']})
    samples = sample_frames(a.video, a.output/'samples', interval=60.)
    initial = json.loads((a.plan/'scene.json').read_text())
    final = json.loads((a.plan/'candidate_world.json').read_text())
    rows = []
    for label, frame, state in [('first', samples[0], initial), ('last', samples[-1], final)]:
        observations = pipeline.detect_frame(frame, a.output/label, 'arrival_boundary')
        expected = expected_actors(state, frame['width'], frame['height'])
        pairs, missing = assign(expected, observations, frame['width'], frame['height'])
        rows.append({'boundary': label, 'frame_id': frame['frame_id'], 'timestamp': frame['timestamp'],
                     'matches': pairs, 'missing': missing, 'observations': [o.to_dict() for o in observations]})
        image = Image.open(frame['path']).convert('RGB')
        draw = ImageDraw.Draw(image)
        for item in expected:
            draw.rectangle(item['bbox'], outline='cyan', width=2)
            draw.text(tuple(item['bbox'][:2]), 'planned '+item['world_id'], fill='cyan')
        for pair in pairs:
            draw.rectangle(pair['observed_bbox'], outline='yellow', width=2)
            draw.text(tuple(pair['observed_bbox'][:2]), 'detected '+pair['world_id'], fill='yellow')
        image.save(a.output/(label+'_overlay.jpg'))
    deltas = []
    for first in rows[0]['matches']:
        last = next((m for m in rows[-1]['matches'] if m['world_id'] == first['world_id']), None)
        if last is None:
            deltas.append({'world_id': first['world_id'], 'status': 'UNOBSERVED', 'reason': 'No endpoint assignment'})
            continue
        def center(box):
            box = np.asarray(box)
            return (box[:2]+box[2:])/2
        observed = center(last['observed_bbox'])-center(first['observed_bbox'])
        planned = center(last['expected_bbox'])-center(first['expected_bbox'])
        deltas.append({'world_id': first['world_id'], 'observed_box_delta_px': observed.tolist(),
                       'planned_box_delta_px': planned.tolist(), 'displacement_error_px': float(np.linalg.norm(observed-planned)),
                       'same_assigned_track': first['track_id'] == last['track_id']})
    result = {'world_modified': False, 'frames': rows, 'boundary_displacements': deltas,
              'limitations': ['Role assignment uses approximate projection and category, not proven identity.',
                'Two-boundary association is not continuous tracking.',
                'Box-center displacement can be caused by pose, occlusion or detector error; it is not a foot-root measurement.',
                'Missing detections are not disappearing world objects. Masks/embeddings are evidence artifacts only.']}
    (a.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'boundary_displacements': deltas, 'matches': [len(r['matches']) for r in rows]}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/default.yaml')
    main(p.parse_args())
