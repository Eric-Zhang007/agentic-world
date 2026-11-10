"""Make a recorded-input trial on the real GTA region, without a model call."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from harness.pi.world_tools import read,save,check_plan
from scripts.serve_controls import Session


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    session=Session(read(args.scene),'player',args.output)
    for tick in range(72):
        keys=['w']
        if 24<=tick<48:keys.append('shift')
        if tick==24:keys.append('space')
        session.step({'keys':keys,'mouse':[1.,.1]})
    exported=session.export();plan=Path(exported['plan']);check_plan(plan)
    result={'plan':str(plan),'video_frames':73,'input_ticks':72,'seconds':3.,
            'endpoint_velocity':session.controller.actor['motion']['velocity'],
            'endpoint_camera_motion':session.controller.scene['camera']['motion'],
            'events':session.controller.events,'world_committed':False}
    if np.linalg.norm(result['endpoint_velocity'])<.1:raise ValueError('Trial unexpectedly stopped')
    save(args.output/'verification.json',result)
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
