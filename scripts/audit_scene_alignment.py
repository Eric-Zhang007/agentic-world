"""Audit coarse static anchors before attributing video differences to camera motion."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.action_planner import json_call
from scripts.evaluate_motion_ablation import project
from whitebox.motion import rotation


def validate_audit(result, scene):
    known={e['id'] for e in scene['entities'] if e['kind'] in {'box','stairs'}}
    anchors=result.get('anchors',[])
    if type(result.get('usable_for_camera_alignment')) is not bool:
        raise ValueError('Alignment audit must provide explicit usability')
    if (len(anchors)!=len(known) or {a.get('world_id') for a in anchors}!=known or
            any(a.get('status') not in {'SUPPORTED','CONTRADICTED','UNOBSERVED'} or
                not isinstance(a.get('evidence'),list) or not a['evidence'] or
                any(not isinstance(e,str) or not e.strip() for e in a['evidence']) for a in anchors)):
        raise ValueError('Alignment audit must assess every static ID exactly once with evidence')
    return result


def audit(config, scene, reference, proxy, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=resume)
    width,height = Image.open(reference).size
    projections = []
    for entity in scene['entities']:
        if entity['kind'] not in {'box','stairs'}:
            continue
        corners = np.array(list(itertools.product([-1,1],repeat=3)))*np.array(entity['size'])/2
        corners = corners @ rotation(entity.get('rotation_y',0.)).T + entity['position']
        pixels = project(corners,scene['camera'],width,height)
        projections.append({'id':entity['id'],
            'projected_bounds_xyxy':np.r_[pixels.min(axis=0),pixels.max(axis=0)].tolist() if pixels is not None else None,
            'scope':'Unoccluded projected coarse volume; not proof of visibility'})
    payload = {'scene':scene,'reference_resolution':[width,height], 'static_projections':projections,
               'image_order':['actual established video boundary','same-time whitebox proxy'],
               'scope':'Audit current representation only; no accepted history is revoked or modified.'}
    if resume and (output/'input.json').exists():
        if json.loads((output/'input.json').read_text()) != payload:
            raise ValueError('Cannot resume alignment audit with changed inputs')
        if (output/'review.json').exists():
            return validate_audit(json.loads((output/'review.json').read_text()),scene)
    else:
        (output/'input.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    result = json_call(config,
        'Audit INITIAL coarse scene alignment before attempting camera reconciliation. '
        'First image is the established real video boundary; second is its current whitebox. '
        'Return {usable_for_camera_alignment:boolean,anchors:[{world_id:string,'
        'status:SUPPORTED|CONTRADICTED|UNOBSERVED,evidence:[string]}],issues:[string],'
        'recommended_next_step:string}. Judge only key STATIC architectural masses, relative locations, '
        'and occlusion order. Ignore texture, ornament, fine mesh, colors and minor geometric errors. '
        'Do not require invisible/offscreen objects to appear. Use supplied world IDs, geometry and '
        'projection bounds to identify whether visible key masses have a plausible counterpart. '
        'A statue can be a plain cuboid, but a clearly visible statue behind an actor cannot be represented '
        'by a cuboid outside the view. Distinguish pre-existing baseline mismatch from later camera motion. '
        'This is an evidence audit, not permission to relocate established scenery or overwrite history. '
        'Assess EVERY ID in static_projections exactly once, including UNOBSERVED when appropriate. '
        'Use ONLY IDs present in static_projections for anchors; put unmatched visual structures '
        'in issues instead of inventing IDs. evidence must be a nonempty list of strings. '
        'Do not automatically endorse the whitebox or demand a detailed reconstruction.',
        payload,[reference,proxy])
    raw_index=len(list(output.glob('raw_response_*.json')))
    (output/f'raw_response_{raw_index:02d}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    validate_audit(result,scene)
    result['world_modified'] = False
    (output/'review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene','reference','proxy','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    a = p.parse_args()
    print(json.dumps(audit(yaml.safe_load(a.config.read_text())['agent'],
        json.loads(a.scene.read_text()),a.reference,a.proxy,a.output),indent=2))
