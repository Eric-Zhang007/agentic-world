"""Recover the exact dataset frame's interpolated screen-ray calibration."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.bundle / 'tools'))
    sys.path.insert(0, str(a.bundle / 'method/source/src'))
    import common
    from wm_engine.processing.client_samples import _interpolate_camera
    from wm_engine.proxy.geometry import Camera
    state = json.loads(a.state.read_text())
    frames = Path(state['source_manifest']).parent / 'frames.jsonl'
    row = next(json.loads(line) for line in frames.open() if json.loads(line)['proxy_frame_index'] == state['frame_index'])
    info = row['camera_interpolation']
    ep = common.episode(state['episode_id'])
    def sample(seq):
        return next(common.spool_rows(ep['spools'][state['agent_id']],
            'seq=? AND client_domain=? AND domain=?',
            [seq, row['client_clock_domain'], row['server_clock_domain']]))['camera']
    low = sample(info['lower_client_sample_seq'])
    high = sample(info['upper_client_sample_seq'])
    camrow = _interpolate_camera(low, high, info['alpha'])
    width, height = state['resolution']
    cam = Camera(camrow, width, height)
    scene = json.loads(a.scene.read_text())
    origin = np.asarray(json.loads((a.scene.parent / 'urban_region/region.json').read_text())['origin_gta'])
    transform = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    position = transform @ (cam.position-origin)
    scene['camera'] = {'position':position.tolist(), 'target':(position+transform@cam.basis[2]).tolist(),
        'up':(-transform@cam.basis[1]).tolist(), 'fov':float(np.degrees(2*np.arctan(height/(2*cam.fy)))),
        'intrinsics':{'width':width,'height':height,'fx':cam.fx,'fy':cam.fy,'cx':width/2,'cy':height/2},
        'calibration_source':cam.source, 'near':cam.near}
    scene['review_note'] = 'Static scene with dataset camera interpolation and calibrated screen rays; dynamic entities omitted.'
    a.output.write_text(json.dumps(scene,indent=2),encoding='utf-8')
    (a.output.parent/'camera_calibration.json').write_text(json.dumps({'camera':scene['camera'],
        'frame_row':row,'source_camera':camrow},indent=2),encoding='utf-8')
    print(json.dumps(scene['camera']))


if __name__ == '__main__':main()
