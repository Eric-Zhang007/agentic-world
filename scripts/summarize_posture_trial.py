"""Audit the pose actually committed, then render existing loop diagnostics."""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from state.db import WorldDB
from whitebox.posture import weights,contacts
from whitebox.world import scene_from_db
from whitebox.motion import simulate
from scripts.summarize_video_reconcile import main as summarize


def actors(scene):
    return {e['id']:{'posture_weights':weights(e),'support_contacts':contacts(e),'grip':e.get('grip','both')}
            for e in scene['entities'] if e['kind']=='actor'}


def main(root):
    summarize(root)
    db=WorldDB(root/'world.sqlite')
    try:
        parent=json.loads((root/'parent_history.json').read_text())
        history=db.get_meta('whitebox_history')
        assert history[:len(parent)]==parent
        records=[]
        for folder in sorted(root.glob('segment_*')):
            if not (folder/'review/status.json').exists():continue
            status=json.loads((folder/'review/status.json').read_text())
            row={'segment':folder.name,'status':status}
            planned=json.loads((folder/'plan/timeline.json').read_text())
            start=json.loads((folder/'plan/status.json').read_text())['world_start']
            replay,_,_=simulate(json.loads((folder/'plan/scene.json').read_text()),
                                json.loads((folder/'plan/action.json').read_text()),start,planned['fps'])
            assert replay==planned
            row['plan_replays_with_deployed_code']=True
            row['planned_endpoint']=actors(json.loads((folder/'plan/candidate_world.json').read_text()))
            if status['status']=='COMMITTED':
                actual=json.loads((folder/'review/world_after.json').read_text())
                pending=json.loads((folder/'review/next_plan/scene.json').read_text())
                assert actual==pending
                row['accepted_endpoint']=actors(actual)
                row['next_plan_initial']=actors(pending)
                row['original_action_complete']=status['action_complete']
                corrected=json.loads((folder/'review/corrected_timeline.json').read_text())
                row['first_low_guard_seconds']={}
                for label,trajectory in [('planned',planned),('accepted',corrected)]:
                    row['first_low_guard_seconds'][label]={identity:next((f['time']-start
                        for f in trajectory['frames'] for entity in f['entities']
                        if entity['id']==identity and weights(entity).get('low_guard',0)>1-1e-8),None)
                        for identity in actors(actual)}
                next_action=json.loads((folder/'review/next_plan/action.json').read_text())
                if all(p['type']=='wait' for p in next_action['phases']):
                    next_timeline=json.loads((folder/'review/next_plan/timeline.json').read_text())
                    expected=actors(actual)
                    for frame in next_timeline['frames']:
                        for entity in frame['entities']:
                            if entity['kind']=='actor':
                                assert weights(entity)==expected[entity['id']]['posture_weights']
                    row['pending_wait_preserves_posture_in_simulation']=True
                final=max((folder/'review/frames').glob('*.png'))
                candidates=[max((folder/'planned_preview').glob('frame_*.png')),final,
                            max((folder/'corrected_preview').glob('frame_*.png'))]
                labels=['Planned endpoint','Actual generated endpoint','Accepted corrected coarse endpoint']
                sheet=Image.new('RGB',(1536,412),'black');draw=ImageDraw.Draw(sheet)
                for i,(path,label) in enumerate(zip(candidates,labels)):
                    sheet.paste(Image.open(path).convert('RGB').resize((512,384)),(512*i,28))
                    draw.text((512*i+8,8),label,fill='white')
                sheet.save(root/(folder.name+'_posture_reconcile.jpg'))
            records.append(row)
        result={'parent_history_unchanged':True,'new_commits':len(history)-len(parent),
                'revision':db.get_meta('whitebox_revision'),'world_time':db.get_meta('tick'),
                'pending_action':db.get_meta('whitebox_video_action'),
                'current_actors':actors(scene_from_db(db)),'segments':records,
                'scope':'Agent-estimated coarse posture and template contacts; not video skeleton reconstruction'}
        (root/'posture_state_report.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root.resolve())
