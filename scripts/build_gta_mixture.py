"""Build portable task manifests and finite balanced rounds, preserving old media."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import orjson

from memory.gta_mixed_dataset import balanced_round


def write_rows(path,rows):
    path.write_bytes(b''.join(orjson.dumps(r)+b'\n' for r in rows))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--objects',type=Path,default=Path('D:/GTA_memory_release_20261018/dataset'))
    parser.add_argument('--scenes',type=Path,default=Path('D:/GTA_scene_pairs_complete_20261019/dataset'))
    parser.add_argument('--output',type=Path,default=Path('D:/GTA_memory_mixture_20261019'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    splits=orjson.loads((args.objects/'acceptance_report.json').read_bytes())['episode_split_map']
    rows=[]
    for split in ('train','validation','test'):
        for line in (args.objects/'splits'/(split+'.jsonl')).read_bytes().splitlines():
            item=orjson.loads(line);sample=orjson.loads((args.objects/item['sample_root']/'sample.json').read_bytes())
            edges=sample['object_references'];categories={e['category'] for e in edges}
            task='object_vehicle' if categories=={'vehicle'} else 'object_person' if 'vehicle' not in categories else 'object_mixed'
            rows.append({'dataset':'objects','sample_id':sample['sample_id'],'sample_root':item['sample_root'],
                'split':split,'episode_id':sample['episode_id'],'task':task,
                'balance_group':'|'.join(sorted(e['object_id'] for e in edges)),
                'categories':sorted(categories),'age_bin':'under_30s','angle_bin':'not_annotated',
                'lighting':'not_annotated','scene':'not_annotated'})
    for line in (args.scenes/'samples.jsonl').read_bytes().splitlines():
        item=orjson.loads(line);sample=orjson.loads((args.scenes/item['sample_root']/'sample.json').read_bytes())
        if sample.get('training_status')!='REVIEWED_SCENE_VIDEO':continue
        label=sample['visual_review'];age=sample['age_sec'];angle=sample['view_angle_deg']
        rows.append({'dataset':'scenes','sample_id':sample['sample_id'],'sample_root':item['sample_root'],
            'split':splits[sample['episode_id']],'episode_id':sample['episode_id'],'task':'scene_background',
            'balance_group':sample['episode_id']+':'+label['site'],'scene':label['scene'],
            'lighting':label['lighting'],'age_bin':'30_60s' if age<60 else '60_180s' if age<180 else 'over_180s',
            'angle_bin':'under_10deg' if angle<10 else '10_25deg' if angle<25 else '25_35deg','revisit_kind':sample['kind']})
    fractions={'scene_background':.5,'object_person':.2,'object_vehicle':.2,'object_mixed':.1}
    report={'suggested_task_fractions':fractions,'max_scene_per_site_per_round':2,'ratios_are_unvalidated_starting_point':True,'splits':{},'rounds':[]}
    for split in ('train','validation','test'):
        part=[r for r in rows if r['split']==split];write_rows(out/(split+'.jsonl'),part)
        report['splits'][split]={key:dict(Counter(r.get(key,'not_annotated') for r in part))
            for key in ('task','scene','lighting','age_bin','angle_bin','episode_id','revisit_kind')}
        report['splits'][split]['scene_site_count']=len({r['balance_group'] for r in part if r['dataset']=='scenes'})
    train=[r for r in rows if r['split']=='train']
    for seed in range(10):
        plan=balanced_round(train,fractions,seed);write_rows(out/f'train_round_{seed:02d}.jsonl',plan)
        report['rounds'].append({'seed':seed,'unique_samples':len(plan),'counts':dict(Counter(r['task'] for r in plan)),
            'background_sites':dict(Counter(r['balance_group'] for r in plan if r['dataset']=='scenes'))})
    report['warnings']=['Small background pool limits a balanced round; rounds repeat scene samples across epochs, not within a round.',
        'Do not interpret frame count or repeated samples as independent scene diversity.',
        'Old object samples have no scene/light/view-angle labels; their absence is reported, not guessed.',
        'Episode split prevents same-trajectory leakage, not shared-map-location leakage across episodes.',
        'Validation may lack scene samples. Do not tune background hyperparameters on the test split.',
        'No H3 training or optimal mixture ratio established.']
    (out/'distribution_report.json').write_bytes(orjson.dumps(report,option=orjson.OPT_INDENT_2))
    (out/'roots.example.json').write_text(json.dumps({'objects':str(args.objects),'scenes':str(args.scenes)},indent=2))
    shutil.copy2('memory/gta_mixed_dataset.py',out/'dataset.py')
    (out/'README.md').write_text('''# Mixed memory training data

Existing object media is preserved and referenced, not duplicated. Extract the
previous object release and new scene release separately. Set roots for `objects`
and `scenes` to those extracted datasets. Manifests use relative sample paths.

Starting task mix: scene background 50%, person 20%, vehicle 20%, mixed objects
10%. This is a small diagnostic training schedule, not a validated optimal ratio.
Each generated round is capped by the available scene pool and has no duplicate
sample, with at most two background examples from the same labeled site.
Object-instance groups and scene sites are cycled before repeating a
group. Different rounds rotate object examples; scene examples repeat across
rounds. Do not run long training on the tiny scene pool merely to consume all
4,984 old examples. `distribution_report.json` reports actual counts, missing
strata, site counts, and simulation results. Full train/validation/test lists
remain available for future sampling experiments. No test samples enter rounds.

```python
from dataset import MixedMemoryDataset
data = MixedMemoryDataset('train_round_00.jsonl',
    {'objects': '/data/objects', 'scenes': '/data/scenes'})
sample = data[0]
# sample['conditions'] and sample['supervision'] are kept separate.
```

Both tasks use a historical RGB/whitebox frame pair, a 120-frame target whitebox
video, and aligned RGB video supervision. Object references are empty for scene
tasks. QA text and future lifecycle fields do not enter conditioning. The loader
can apply explicit common temporal_indices to RGB, whitebox and object boxes;
model-specific 4n+1 temporal adaptation is the trainer's responsibility. No
default temporal stretching or duplicated frame is applied. Dependencies:
NumPy and OpenCV. No training is launched by these scripts.
''',encoding='utf-8')
    print(orjson.dumps({'counts':dict(Counter(r['task'] for r in rows)),'round0':report['rounds'][0]},option=orjson.OPT_INDENT_2).decode())


if __name__=='__main__':main()
