"""Build verified lossless cadence repairs and corrected delivery manifests."""
import csv
import io
import json
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from verify_reaudit_pts_repair import candidate, compare

AUDIT=Path(__file__).resolve().parents[1]/'outputs/dataset_reaudit'
ARCHIVE=Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip')
OUT=Path('D:/GTA_memory_patch')
TIMING='encoded_media_timing_or_decode_issue'

def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def repair(sid):
    path=OUT/'verification'/f'{sid}.json'
    if path.exists():
        return json.loads(path.read_text())
    result={'sample_id':sid,'files':[]}
    with zipfile.ZipFile(ARCHIVE) as z:
        base=f'samples/{sid}'
        sample=json.loads(z.read(base+'/sample.json'))
        for name in ('target_rgb','target_whitebox'):
            member=base+'/'+name+'.mp4'
            data=z.read(member)
            destination=OUT/'payload'/member
            destination.parent.mkdir(parents=True,exist_ok=True)
            candidate(data,destination,sample['fps'])
            checked=compare(data,destination,sample['frame_count'],sample['fps'])
            checked.pop('timestamps')
            result['files'].append({'path':member,'bytes':destination.stat().st_size,**checked})
    path.write_text(json.dumps(result,indent=2))
    return result

def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'verification').mkdir(exist_ok=True)
    ledger=read_rows(AUDIT/'review_ledger.jsonl')
    metadata={r['sample_id']:r for r in read_rows(AUDIT/'metadata/records.jsonl')}
    flagged_sources={tuple(r['source']) for r in read_rows(AUDIT/'metadata/source_clock_rates.jsonl') if r.get('flagged')}
    excluded={r['sample_id']:list(r['reasons']) for r in ledger if any(s!=TIMING for s in r['reasons'])}
    # A faulty source used as history also contaminates an otherwise normal target.
    for r in ledger:
        bad=[ref['source'] for ref in metadata[r['sample_id']]['reference_sources'] if tuple(ref['source']) in flagged_sources]
        if bad:
            excluded.setdefault(r['sample_id'],[]).append('reference_uses_flagged_clock_source')
    repaired=[r['sample_id'] for r in ledger if r['reasons']==[TIMING] and r['sample_id'] not in excluded]
    plan={'original_records':len(ledger),'repair_records':len(repaired),'exclude_records':len(excluded),'remaining':len(ledger)-len(excluded)}
    (OUT/'progress.json').write_text(json.dumps({**plan,'completed_repairs':0,'state':'running'},indent=2))
    print(json.dumps(plan),flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(repair,sid):sid for sid in repaired}
        results=[]
        for future in as_completed(futures):
            results.append(future.result())
            (OUT/'progress.json').write_text(json.dumps({**plan,'completed_repairs':len(results),'state':'running'},indent=2))
            print(f'Repaired and pixel-verified {len(results)}/{len(repaired)}',flush=True)
    active={r['sample_id'] for r in ledger}-set(excluded)
    manifest_info={}
    originals={}
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in z.namelist():
            if name.startswith('manifests/') and name.endswith('.jsonl'):
                raw=z.read(name)
                rows=[json.loads(s) for s in raw.decode('utf-8').splitlines() if s.strip()]
                originals[name]=rows
                new=[r for r in rows if r['sample_id'] in active]
                dest=OUT/'payload'/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new),encoding='utf-8')
                manifest_info[name]={'before':len(rows),'after':len(new)}
        report=json.loads(z.read('DATASET_REPORT.json'))
    base_rows=[r for name in ('manifests/train.jsonl','manifests/validation.jsonl','manifests/test.jsonl') for r in originals[name] if r['sample_id'] in active]
    assert len(base_rows)==len(active)
    assert {r['sample_id'] for r in base_rows}==active
    report.update(records=len(base_rows),tasks=dict(Counter(r['task'] for r in base_rows)),splits=dict(Counter(r['split'] for r in base_rows)),source_release_counts=dict(Counter(r['source_release'] for r in base_rows)))
    windows=Counter((tuple(metadata[r['sample_id']]['source']),metadata[r['sample_id']]['target_start'],metadata[r['sample_id']]['target_end_exclusive']) for r in base_rows)
    report.update(unique_target_windows=len(windows),reused_target_windows=len(base_rows)-len(windows))
    report['audit_patch']={'cadence_repaired':len(repaired),'excluded':len(excluded),'scope':'Lossless CFR repair plus exclusion of unresolved audit flags; not full visual certification'}
    (OUT/'payload/DATASET_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for r in ledger:
        sid=r['sample_id']
        if r['reasons'] or sid in excluded:
            rows.append({'sample_id':sid,'split':r['split'],'task':r['task'],'action':'EXCLUDE' if sid in excluded else 'REPAIR_CFR','reasons':excluded.get(sid,r['reasons'])})
    (OUT/'issues.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    with (OUT/'issues.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['sample_id','split','task','action','reasons']);writer.writeheader()
        writer.writerows({**r,'reasons':';'.join(r['reasons'])} for r in rows)
    (OUT/'excluded_sample_ids.txt').write_text('\n'.join(sorted(excluded))+'\n')
    shutil.copy2(AUDIT/'normal_random_review/review.json',OUT/'visual_spotcheck_advisory.json')
    files=[{'path':p.relative_to(OUT/'payload').as_posix(),'bytes':p.stat().st_size} for p in sorted((OUT/'payload').rglob('*')) if p.is_file()]
    spec={**plan,'source_archive_bytes':ARCHIVE.stat().st_size,'source_archive_name':ARCHIVE.name,'repair_ids':repaired,'excluded_ids':sorted(excluded),'manifests':manifest_info,'files':files,'original_manifest_records':originals,'summary':report}
    (OUT/'patch.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('apply_delivery_patch.py'),OUT/'apply_patch.py')
    doc=f'''# GTA memory 训练集修正补丁

适用：后一次完整交付的 GTA_memory_training_unified.zip，原包 {ARCHIVE.stat().st_size:,} 字节、7085条记录。

## 改了什么

- {len(repaired)}条视频时间戳修复：按原解码帧序设置为24fps，无损编码。两路视频每一帧解码RGB像素都与原件比较相同，帧数不变。
- {len(excluded)}条暂时移出训练索引：采集时间缺口、时钟比例异常、已知对齐问题，以及历史参考来自异常时钟源的样本。不是把这些问题假装修好。
- 剩余{len(active)}条，训练/验证/测试与全部train_round索引同步更新；任务和来源计数更新。
- 样本文件夹不删除，原索引和被替换视频会备份。源数据、历史帧、白模世界状态不擅自改写。

## 怎么用

解压本补丁，然后对已经解压的完整数据集根目录运行（根目录应有samples、manifests、dataset.py）：

```bash
python apply_patch.py --root /path/to/GTA_memory_training --check
python apply_patch.py --root /path/to/GTA_memory_training --apply
```

只需要Python标准库，无需安装视频模型。脚本核对原始索引内容、修复对象和补丁文件尺寸，备份后复制替换。重复运行会验证已经应用的补丁。备份位于数据根目录的patch_backup目录。

训练继续使用manifests/train.jsonl或原来的train_round_XX.jsonl。不要绕过索引扫描samples目录，否则被剔除样本仍会读入。自行写dataloader时，可直接使用excluded_sample_ids.txt过滤。

## 清单

- issues.csv / issues.jsonl：逐条操作和原因。REPAIR_CFR为视频时间戳修复，EXCLUDE为从训练索引移除。
- verification/：每条修复的逐帧像素和时间戳检查结果。
- visual_spotcheck_advisory.json：24条随机视觉抽查记录。9条待进一步查看，不作为程序已确诊错帧批量删除。
- patch.json：完整操作、文件和数量记录。

这次修正的是已发现的问题，并不是全包视觉验收。时间戳修复不改变RGB与白模的源时间对应关系；涉及源时间映射的问题先移除，不能靠重编码修复。
'''
    (OUT/'README.md').write_text(doc,encoding='utf-8')
    (OUT/'progress.json').write_text(json.dumps({**plan,'completed_repairs':len(results),'state':'built'},indent=2))
    print(json.dumps({'state':'built',**plan,'payload_bytes':sum(f['bytes'] for f in files)}),flush=True)

if __name__=='__main__':
    main()
