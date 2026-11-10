"""Replay the actual optional-NPC rejection on an isolated world, without API/H3."""
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness/pi'))
from world_tools import dispatch,read


def main():
    base=ROOT/'outputs/gta_pi_loop_20261102/rural'
    actual=base/'continuation'
    review=read(actual/'commit/review.json')
    with tempfile.TemporaryDirectory(prefix='npc-motion-replay-') as temporary:
        folder=Path(temporary);db=str(folder/'world.sqlite')
        dispatch('fork',{'source':str(base/'world.sqlite'),'db':db})
        before=dispatch('status',{'db':db})
        dispatch('plan',{'db':db,'action':str(actual/'action.json'),'output':str(folder/'plan')})
        dispatch('inspect_video',{'plan':str(folder/'plan'),'video':str(actual/'render/controlled.mp4'),
            'output':str(folder/'evidence')})
        result=dispatch('commit',{'db':db,'plan':str(folder/'plan'),'evidence':str(folder/'evidence/evidence.json'),
            'review':str(actual/'commit/review.json'),'output':str(folder/'accepted')})
        after=dispatch('status',{'db':db})
        source_after=dispatch('status',{'db':str(base/'world.sqlite')})
        assert before==source_after
        assert result['status']=='COMMITTED' and after['revision']==before['revision']+1
        npc=next(e for e in after['scene']['entities'] if e['id']=='npc_ahead')
        assert npc['motion']['provenance']=='AGENT_ACCEPTED_VIDEO_ESTIMATE'
        report={'scope':'Replay of actual rejected review on temporary fork; no new video/model call',
            'source_revision_unchanged':source_after['revision'],
            'replay_revision':after['revision'],'result':result,
            'review_motion_ids':[m['id'] for m in review['motion_assessments']],
            'npc_endpoint_motion':npc['motion']}
        (base.parent/'npc_motion_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
