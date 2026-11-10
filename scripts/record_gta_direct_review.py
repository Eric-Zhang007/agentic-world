"""Record explicit local visual decisions. No network or model calls."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    batch=a.root/'pending_audit_full_source'
    keys=json.loads((batch/'index.json').read_bytes())
    groups=json.loads(a.decisions.read_text())
    seen=set();records=[]
    for group in groups:
        for index in group['indices']:
            if type(index) is not int or not 0<=index<len(keys) or index in seen:
                raise ValueError('Invalid or duplicate index')
            seen.add(index)
            if not (batch/f'sheet_{index//4:03d}.jpg').is_file():raise ValueError('Missing inspected sheet')
            rejection=group.get('rejected',{}).get(str(index))
            row=dict(sample_id=keys[index],alignment_ok=True,background_memory_useful=True,
                no_major_occlusion_conflict=True,no_ui_contamination=True,uncertain=False,
                background_evidence=group['background_evidence'],reason=group['reason'])
            if rejection:
                field,reason=rejection
                if field not in ('alignment_ok','background_memory_useful','no_major_occlusion_conflict','no_ui_contamination','uncertain'):
                    raise ValueError('Invalid rejection field')
                row[field]=field=='uncertain';row['reason']=reason
            row.update(accepted=rejection is None,method='Direct Codex visual inspection; no external review API',
                sample_path=str(a.root/'training/samples'/keys[index]/'sample.json'),
                sheet=str(batch/f'sheet_{index//4:03d}.jpg'),sheet_index=index,
                decision_source=str(a.decisions),reviewed_at=time.time(),
                sampled_target_frames=[0,30,60,90,119],
                scope='Historical pair and five target pairs; sampled inspection, not all-frame temporal validation')
            records.append(row)
    a.output.mkdir(parents=True,exist_ok=True)
    for row in records:
        path=a.output/(row['sample_id']+'.json')
        if path.exists():raise FileExistsError('Preserve earlier direct review: '+str(path))
    for row in records:
        (a.output/(row['sample_id']+'.json')).write_text(json.dumps(row,indent=2))
    print(json.dumps({'reviewed':len(records),'accepted':sum(r['accepted'] for r in records),
                      'rejected':sum(not r['accepted'] for r in records)}))


if __name__=='__main__':main()
