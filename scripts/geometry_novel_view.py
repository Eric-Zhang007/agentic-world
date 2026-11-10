"""Re-render a frozen candidate scene from a registered observed camera, without merging."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/geometry_init/urban')
    args = p.parse_args(); root = args.root
    scene = json.loads((root/'coarse_assets/scene.json').read_text())
    original_entities = deepcopy(scene['entities'])
    mapping = json.loads((root/'coarse_assets/report.json').read_text())
    with np.load(root/'alignment/registration.npz') as registration:
        if not registration['accepted'].item():
            raise ValueError('Registration failed; no diagnostic camera can be transferred')
        matrix = np.asarray(mapping['model_to_scene']) @ registration['transform']
    with np.load(root/'reconstruction/joint.npz') as data:
        e, k = data['extrinsic'][0], data['intrinsic'][0]
        height, width = data['images'].shape[-2:]
    r, t = e[:, :3], e[:, 3]
    position = matrix[:3, :3] @ (-r.T @ t) + matrix[:3, 3]
    forward, up = matrix[:3, :3] @ r.T[:, 2], matrix[:3, :3] @ -r.T[:, 1]
    scene['camera'] = {'position': position.tolist(),
                       'target': (position+forward/np.linalg.norm(forward)).tolist(),
                       'up': (up/np.linalg.norm(up)).tolist(),
                       'fov': math.degrees(2*math.atan(height/(2*k[1, 1]))),
                       'intrinsics': {'width': width, 'height': height, 'fx': float(k[0, 0]),
                                      'fy': float(k[1, 1]), 'cx': float(k[0, 2]), 'cy': float(k[1, 2])}}
    scene['static_geometry'] = {'path': '../coarse_assets/ground.bin',
                                'collision_path': '../coarse_assets/ground_collision.bin'}
    assert scene['entities'] == original_entities
    output = root/'novel_view'; output.mkdir(exist_ok=True)
    (output/'scene.json').write_text(json.dumps(scene, indent=2), encoding='utf-8')
    (output/'report.json').write_text(json.dumps({
        'purpose': 'Frozen old geometry rendered from estimated historical reference camera',
        'new_geometry_merged': False, 'entities_unchanged': True,
        'dynamic_objects': 'Frozen at target time; do not compare their historical positions',
        'world_merge_ready': False}, indent=2), encoding='utf-8')
    print(output/'scene.json')


if __name__ == '__main__':
    main()
