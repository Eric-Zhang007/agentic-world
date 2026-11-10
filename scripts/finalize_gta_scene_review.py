"""Apply the explicit visual decisions recorded during the scene-pair session."""
from pathlib import Path
from collections import Counter
import orjson


def main():
    root=Path('D:/GTA_scene_pairs_complete_20261019');dataset=root/'dataset'
    extra={'scene_pair_00037':'Later target views contain a large proxy plane/occlusion unlike the RGB airplane foreground.',
           'scene_pair_00096':'Construction-ramp target develops major proxy slabs/occlusion at the middle sampled frames.'}
    rows=[]
    for p in sorted((dataset/'samples').glob('*/sample.json')):
        sample=orjson.loads(p.read_bytes())
        if sample.get('training_status')=='TARGET_VIDEO_AWAITING_VISUAL_REVIEW':
            sample['training_status']='DIAGNOSTIC_ONLY' if sample['sample_id'] in extra else 'REVIEWED_SCENE_VIDEO'
            if sample['sample_id'] in extra:sample['exclusion_reason']=extra[sample['sample_id']]
            sample['video_visual_review']={'reviewer':'Codex visual inspection in this session',
                'frames':[0,30,60,90,119],'scope':'RGB/whitebox contact sheets, not all frames visually inspected',
                'accepted':sample['sample_id'] not in extra}
            p.write_bytes(orjson.dumps(sample,option=orjson.OPT_INDENT_2))
        rows.append({'sample_id':sample['sample_id'],'status':sample['training_status'],
                     'reason':sample.get('exclusion_reason')})
    (dataset/'training_review.json').write_bytes(orjson.dumps(rows,option=orjson.OPT_INDENT_2))
    print(dict(Counter(r['status'] for r in rows)))


if __name__=='__main__':main()
