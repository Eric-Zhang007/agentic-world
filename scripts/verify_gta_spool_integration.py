"""Verify original projection code against all newly indexed paired batches."""
from pathlib import Path
import argparse
import orjson
from memory.gta_release import Records,frame_evidence


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('D:/GTA_spool_integration_20261020'));args=p.parse_args()
    reports=orjson.loads((args.root/'report.json').read_bytes());results=[];ledger=[]
    for report in reports:
        source=Path(report['root']);records=Records(report['index']);verified=None
        manifests=sorted(source.glob('derived/segments/*/*/manifest.json'))
        for path in manifests:
            m=orjson.loads(path.read_bytes())
            ledger.append({'source_manifest':str(path),'episode_root':str(source),
                           'record_index':Path(report['index']).name,'episode_id':m['episode_id'],
                           'agent_id':m['agent_id'],'segment':m['segment'],'frame_count':m['frame_count'],
                           'status':'RAW_CAMERA_WORLD_INDEXED_NOT_TRAINING_QUALIFIED',
                           'lifecycle_status':'NOT_ADAPTED','static_visibility_status':'NOT_LABELLED'})
            if verified:continue
            with (path.parent/'frames.jsonl').open('rb') as f:
                for line in f:
                    frame=orjson.loads(line)
                    if not frame.get('camera_interpolation'):continue
                    evidence=frame_evidence(records,m,frame)
                    verified={'manifest':str(path),'frame':frame['proxy_frame_index'],
                              'projected_dynamic_candidates':len(evidence['candidates']),
                              'camera_origin':evidence['camera_origin']}
                    break
        records.close()
        if manifests and verified is None:raise ValueError('No real projection verified')
        results.append({'source':str(source),'projection':verified,'paired_segments':len(manifests)})
    (args.root/'source_ledger.jsonl').write_bytes(b''.join(orjson.dumps(v)+b'\n' for v in ledger))
    (args.root/'integration_verification.json').write_bytes(orjson.dumps({'results':results,'paired_segments':len(ledger),
        'scope':'Source integration and real geometry API checks; not a new training release'},option=orjson.OPT_INDENT_2))
    print(results)


if __name__=='__main__':main()
