"""Run existing mask/projection/visibility checks across explicit scene cases."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_mask_alignment import diagnose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cases = json.loads(a.cases.read_text())['cases']
    identities = [case['id'] for case in cases]
    if not cases or len(set(identities)) != len(identities):
        raise ValueError('Nonempty uniquely named cases required')
    a.output.mkdir(parents=True, exist_ok=False)
    summaries, panels = [], []
    for index, case in enumerate(cases):
        scene = json.loads(Path(case['scene']).read_text())
        bundle = Path(case['perception'])
        perception = json.loads(bundle.read_text())
        started = time.perf_counter()
        report, overlay = diagnose(scene, perception, case['mapping'], bundle.parent)
        report.update(case_id=case['id'], source_scene=case['scene'],
                      source_perception=str(bundle), diagnostic_seconds=time.perf_counter()-started,
                      perception_invoked=False, scope='Explicit mapped regions only; unreviewed is not missing. No automatic acceptance.')
        folder = a.output/f'{index:02d}'
        folder.mkdir()
        (folder/'report.json').write_text(json.dumps(report, indent=2))
        cv2.imwrite(str(folder/'overlay.jpg'), overlay)
        width = 960
        panel = cv2.resize(overlay, (width, round(overlay.shape[0]*width/overlay.shape[1])))
        band = np.full((50,width,3), 25, dtype=np.uint8)
        c = report['counts']
        label = f"{case['id']}: reviewed {c['reviewed_world_entities']}/{c['world_entities']} entities; {report['diagnostic_seconds']:.3f}s"
        cv2.putText(band,label,(10,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        panels.append(np.concatenate([band,panel]))
        rows = []
        for row in report['results']:
            rows.append({**row,'visibility':{k:v for k,v in row['visibility'].items() if k!='blocked_image_pixels'}})
        summaries.append({**report, 'results':rows})
    cv2.imwrite(str(a.output/'comparison.jpg'),np.concatenate(panels))
    (a.output/'report.json').write_text(json.dumps({'cases':summaries,'world_committed':False},indent=2))
    print(json.dumps([{'id':r['case_id'],'counts':r['counts'],'seconds':r['diagnostic_seconds'],
                       'geometry_errors':r['geometry_errors']} for r in summaries],indent=2))


if __name__ == '__main__':
    main()
