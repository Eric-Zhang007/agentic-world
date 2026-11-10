"""Audit actual agent plans and compiled controls without accepting any video."""
import json
import sys
from pathlib import Path
import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness/pi'))
from world_tools import dispatch,check_plan,read
from whitebox.posture import body_size


def main():
    root=ROOT/'outputs/semantic_actions'
    result=read(root/'result.json');report={}
    for name,entry in result['cases'].items():
        scene,timeline=check_plan(Path(entry['plan']))
        state=dispatch('status',{'db':str(root/name/'world.sqlite')})
        assert state['revision']==0 and state['tick']==0 and state['history_length']==0
        path=root/name/'compiled'
        compiled=read(path/'compiled_plan.json')
        assert compiled['scene']==scene and compiled['timeline']==timeline
        with av.open(str(path/'whitebox.mp4')) as video:
            frames=list(video.decode(video=0))
        assert len(frames)==len(timeline['frames'])+1==73
        actors=[next(e for e in frame['entities'] if e['id']=='player') for frame in timeline['frames']]
        airborne=[i for i,e in enumerate(actors) if e.get('support_state')=='airborne']
        report[name]={'agent_plan':entry['plan'],'video_frames':len(frames),
            'minimum_body_height':min(body_size(e)[1] for e in actors),
            'airborne_frames':len(airborne),'endpoint':actors[-1]['position'],
            'world_revision':state['revision'],'history_length':state['history_length'],
            'visual_artifact':str(path/'review.jpg'), 'scope':'planned whitebox; no generated-video acceptance'}
        if airborne:
            landing=max(airborne)+1
            report[name]['landing_velocity']=actors[landing]['motion']['velocity']
            report[name]['next_frame_velocity']=actors[landing+1]['motion']['velocity']
            assert actors[landing]['support_state']=='grounded'
            assert actors[landing]['motion']['velocity'][0]>0
            assert actors[landing+1]['position'][0]>actors[landing]['position'][0]
    framed=root/'jump_framed'
    if (framed/'compiled/compiled_plan.json').is_file():
        scene,timeline=check_plan(framed/'plan')
        original=read(Path(result['cases']['jump']['plan'])/'timeline.json')
        assert [f['entities'] for f in timeline['frames']]==[f['entities'] for f in original['frames']]
        c=scene['camera'];eye=np.array(c['position']);forward=np.array(c['target'])-eye
        forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
        up=np.cross(right,forward);focal=432/(2*np.tan(np.radians(c['fov'])/2))
        pixels=[]
        for frame in timeline['frames']:
            actor=next(e for e in frame['entities'] if e['id']=='player');w,h,d=body_size(actor)
            for x in (-w/2,w/2):
                for y in (0,h):
                    for z in (-d/2,d/2):
                        offset=np.array(actor['position'])+[x,y,z]-eye
                        depth=offset@forward
                        assert depth>0
                        pixels.append([384+focal*(offset@right)/depth,216-focal*(offset@up)/depth])
        pixels=np.asarray(pixels)
        assert np.all(pixels>0) and np.all(pixels<[768,432])
        report['jump_framed']={'same_entity_trajectories':True,'body_projection_min':pixels.min(axis=0).tolist(),
            'body_projection_max':pixels.max(axis=0).tolist(),'camera_adjustment':'test operator, not agent',
            'video':str(framed/'compiled/whitebox.mp4')}
    (root/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
