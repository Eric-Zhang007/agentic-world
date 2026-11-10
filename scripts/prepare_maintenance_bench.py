"""Select same-camera chronological observations; keep GT evaluator-side."""
import json
from pathlib import Path
import shutil
import argparse

def same_stream(initial, value):
    keys = ('episode_id', 'agent_id', 'segment', 'server_clock_domain', 'client_clock_domain')
    return all(initial.get(k) is not None and initial.get(k) == value.get(k) for k in keys)

root=Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bench', type=Path, default=root/'outputs/construction_bench_20261104')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
bench=args.bench
rows=json.loads((bench/'manifest.json').read_text())
bench=args.output or bench
result=[]
for row in rows:
    initial=json.loads(Path(row['state']).read_text())
    samples=Path(row['sample']).parent.parent
    sequence=[]
    prefix=f"{initial['episode_id']}_{initial['agent_id']}_{initial['segment']}_f"
    for folder in samples.glob(prefix+'*'):
        state=folder/'target_state.json'
        if not state.is_file():continue
        value=json.loads(state.read_text())
        if value['time_ns']<=initial['time_ns']:continue
        if not same_stream(initial, value):continue
        if not all((folder/f).is_file() for f in ('target_rgb.png','target_whitebox.png')):continue
        sequence.append((value['time_ns'],folder,value))
    sequence.sort(key=lambda item:item[0])
    steps=[]
    seen=set()
    sequence=[item for item in sequence if item[0] not in seen and not seen.add(item[0])]
    for i,(timestamp,folder,state) in enumerate(sequence[:4],1):
        name=f'step_{i:02d}'
        public=bench/'maintenance_inputs'/row['case']/name
        private=bench/'evaluation_only'/row['case']/name
        public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/'target_rgb.png',public/'input.png')
        shutil.copy2(folder/'target_whitebox.png',private/'whitebox.png')
        shutil.copy2(folder/'target_state.json',private/'state.json')
        elapsed=(timestamp-initial['time_ns'])/1e9
        (public/'observation.json').write_text(json.dumps({'elapsed_sec':elapsed}))
        steps.append(dict(step=name,elapsed_sec=elapsed,source=str(folder)))
    result.append(dict(case=row['case'],steps=steps,status='staged_not_evaluated' if steps else 'no_later_observations'))
(bench/'maintenance_manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
