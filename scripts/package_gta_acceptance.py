"""Create a bounded, proportioned acceptance subset with complete samples."""
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import tempfile
import zipfile

from memory.gta_mixed_dataset import MixedMemoryDataset


def main():
    base=Path('E:/GTA_memory_unified_20261021')
    source=base/'GTA_memory_training_unified.zip'
    dest=base/'GTA_memory_acceptance_under1GB.zip'
    if dest.exists():raise FileExistsError(dest)
    meta=base/'package_metadata'
    rows=[json.loads(x) for x in (meta/'manifests/train.jsonl').read_text().splitlines()]
    rng=random.Random(20261022)
    rng.shuffle(rows)
    groups={}
    for r in rows:
        groups.setdefault(r['task'],[]).append(r)
    # Cycle through source episodes, reference kinds and spatial/instance groups.
    ordered={}
    for task,pool in groups.items():
        bins={}
        for r in pool:bins.setdefault((r['episode_id'],r.get('kind'),r['balance_group']),[]).append(r)
        keys=list(bins);rng.shuffle(keys)
        ordered[task]=[]
        while any(bins.values()):
            for key in keys:
                if bins[key]:ordered[task].append(bins[key].pop())
    with zipfile.ZipFile(source) as src:
        infos=src.infolist()
        sizes=Counter()
        for info in infos:
            parts=info.filename.split('/')
            if parts[0]=='samples':sizes[parts[1]]+=info.compress_size
        for count in range(60,9,-10):
            chosen=[]
            for task,n in [('scene_background',count//2),('object_person',count//5),('object_vehicle',count//5),('object_mixed',count//10)]:
                if task=='scene_background':
                    # Represent all three background relations; continuous history <= half.
                    allocation={'cross_camera_passage':n//2,'same_camera_return':max(1,n//5)}
                    allocation['nearby_history']=n-sum(allocation.values())
                    for kind,k in allocation.items():chosen.extend([r for r in ordered[task] if r['kind']==kind][:k])
                else:chosen.extend(ordered[task][:n])
            if sum(sizes[r['sample_id']] for r in chosen)<900_000_000:break
        assert len(chosen)==count
        ids={r['sample_id'] for r in chosen}
        report={'purpose':'acceptance subset, not an independent benchmark','records':len(chosen),
                'tasks':dict(Counter(r['task'] for r in chosen)),
                'background_kinds':dict(Counter(r['kind'] for r in chosen if r['task']=='scene_background')),
                'episodes':dict(Counter(r['episode_id'] for r in chosen)),
                'seed':20261022,'selection':'Training split only; deterministic episode/kind/site cycling, size budget applied to total count; complete samples',
                'training_run':False}
        with tempfile.TemporaryDirectory(prefix='gta_acceptance_',dir=base) as tmp:
            tmp=Path(tmp)
            (tmp/'manifests').mkdir()
            for r in chosen:r['dataset']='data'
            (tmp/'manifests/acceptance.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in chosen),encoding='utf-8')
            for name in ('dataset_core.py','dataset.py','DATA_FORMAT.md','requirements.txt'):
                shutil.copy2(meta/name,tmp/name)
            (tmp/'example.py').write_text("from pathlib import Path\nfrom dataset import load_dataset\nds=load_dataset(Path(__file__).resolve().parent,'acceptance',[0,30,60,90,119])\ns=ds[0]\nprint(len(ds),s['sample_id'],s['conditions']['target_whitebox'].shape,s['supervision']['target_rgb'].shape)\n",encoding='utf-8')
            readme=Path('docs/gta_unified_delivery/ACCEPTANCE_zh.md').read_text(encoding='utf-8')
            (tmp/'README_zh.md').write_text(readme,encoding='utf-8')
            for info in infos:
                parts=info.filename.split('/')
                if parts[0]=='samples' and parts[1] in ids:src.extract(info,tmp)
            ds=MixedMemoryDataset(tmp/'manifests/acceptance.jsonl',{'data':tmp},[0,30,60,90,119])
            for i in range(len(ds)):
                sample=ds[i]
                assert sample['conditions']['target_whitebox'].shape==sample['supervision']['target_rgb'].shape
                assert 'target_rgb' not in sample['conditions']
                print(f'VALIDATED {i+1}/{len(ds)}',flush=True)
            report['all_samples_loader_passed']=True
            (tmp/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            partial=dest.with_suffix('.partial')
            with zipfile.ZipFile(partial,'w',allowZip64=True) as z:
                for p in sorted(tmp.rglob('*')):
                    if not p.is_file():continue
                    method=zipfile.ZIP_DEFLATED if p.suffix in ('.json','.jsonl','.txt','.md','.py') else zipfile.ZIP_STORED
                    z.write(p,p.relative_to(tmp).as_posix(),compress_type=method,compresslevel=1 if method==zipfile.ZIP_DEFLATED else None)
            assert partial.stat().st_size<1_000_000_000
            with zipfile.ZipFile(partial) as z:assert z.testzip() is None
            partial.rename(dest)
    report.update(archive=str(dest),bytes=dest.stat().st_size,archive_crc_passed=True)
    (base/'ACCEPTANCE_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
