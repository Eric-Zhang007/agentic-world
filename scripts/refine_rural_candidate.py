"""Explicit source-view edits for the rural diagnostic; not a general fitter."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
scene = json.loads(a.source.read_text())
a.output.mkdir(parents=True, exist_ok=False)
(a.output/'scene_before.json').write_text(json.dumps(scene, indent=2))
# Source RGB and prior camera preview show these landmark displacements.
# All values remain single-view estimates, not measured metric reconstruction.
changes = {
    'near_utility_pole': {'position':[4.3,4,-5], 'size':[0.18,8,0.18]},
    'center_shop': {'position':[-3.5,2.35,-40], 'size':[10,4.7,6]},
    'center_shop_bays': {'position':[-3.5,1.65,-36.94], 'size':[9,3.3,0.1]},
    'palm_left_trunk': {'position':[-3.6,5,-20], 'size':[0.5,10,0.5]},
    'palm_left_crown': {'position':[-3.6,10,-20]},
    'palm_right_trunk': {'position':[7.3,4.5,-20], 'size':[0.5,9,0.5]},
    'palm_right_crown': {'position':[7.3,9,-20]},
    'roadside_shrubs': {'size':[25,0.25,1.2], 'position':[-4,0.125,-18.6]},
    'near_right_shrubs': {'position':[5.8,0.3,-5.5], 'size':[4,0.6,3]},
    'concrete_apron': {'size':[7,0.02,7.5]},
}
for entity in scene['entities']:
    entity.update(changes.get(entity['id'], {}))
    if entity['kind'] == 'actor':
        entity['asset_id'] = 'gta_human'
(a.output/'scene_after.json').write_text(json.dumps(scene, indent=2))
(a.output/'edits.json').write_text(json.dumps({
    'method':'Codex visual source-view correction, first candidate only',
    'geometry_ready':False, 'changes':changes,
    'limitations':['Not calibrated; needs rendered review',
                   'VGGT candidate inspected but not adopted as metric truth',
                   'No accepted world or history modified']}, indent=2))
