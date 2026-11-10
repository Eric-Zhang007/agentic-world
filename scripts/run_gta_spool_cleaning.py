"""Run remaining passage mining/export after source validation completes."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import orjson


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('D:/GTA_spool_passages_20261020'));a=p.parse_args()
    while not (a.root/'preparation_report.json').exists():time.sleep(10)
    jobs=[('survey','scripts.survey_gta_passage_pairs',['--index',str(a.root/'trajectory_points.jsonl'),'--output',str(a.root)]),
          ('match','scripts.match_gta_passage_pairs',['--root',str(a.root)]),
          ('export','scripts.export_gta_passage_training',['--root',str(a.root),'--release',str(a.root),'--workers','3'])]
    for stage,module,arguments in jobs:
        (a.root/'pipeline_status.json').write_bytes(orjson.dumps({'stage':stage,'status':'RUNNING'}))
        with (a.root/(stage+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u','-m',module,*arguments],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            (a.root/'pipeline_status.json').write_bytes(orjson.dumps({'stage':stage,'status':'FAILED','exit_code':result.returncode}))
            raise RuntimeError(stage+' failed; inspect stage log')
        print(stage,'COMPLETE',flush=True)
    (a.root/'pipeline_status.json').write_bytes(orjson.dumps({'stage':'visual_review','status':'AWAITING_REVIEW'}))


if __name__=='__main__':main()
