"""Validate every exported sample, enforce episode splits, and archive release."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import zipfile
import subprocess
import zlib

import cv2
import numpy as np
import av

from memory.gta_training_dataset import safe_path,read_video,GTAObjectMemoryDataset
from memory.gta_conditioning import reference_prompt

VALIDATOR_VERSION=5


def write_archive(destination, files, extra=None):
    """Publish only complete ZIPs; reuse an existing ZIP after content validation."""
    files=list(files)
    extra=extra or {}
    expected={name:path.stat().st_size for path,name in files}
    expected.update({name:len(value.encode('utf-8')) for name,value in extra.items()})
    if len(expected)!=len(files)+len(extra):raise ValueError('Duplicate archive members')
    def verify(path):
        with zipfile.ZipFile(path) as archive:
            members=archive.infolist()
            if len(members)!=len(expected) or {m.filename:m.file_size for m in members}!=expected:
                raise ValueError('Existing archive does not match release: '+str(path))
            if archive.testzip():raise ValueError('Archive corruption: '+str(path))
            # A repaired video can have the same size as its previous export.
            for source,name in files:
                crc=0
                with source.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):crc=zlib.crc32(block,crc)
                if archive.getinfo(name).CRC!=crc:raise ValueError('Stale archive content: '+name)
            for name,value in extra.items():
                if archive.read(name)!=value.encode('utf-8'):raise ValueError('Stale archive metadata: '+name)
    if destination.exists():
        verify(destination)
        return
    partial=destination.with_name(destination.name+'.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_STORED,allowZip64=True) as archive:
        for source,name in files:
            compression=zipfile.ZIP_DEFLATED if source.suffix in ('.json','.jsonl','.txt','.md','.py') else zipfile.ZIP_STORED
            archive.write(source,name,compress_type=compression,compresslevel=1)
        for name,value in extra.items():archive.writestr(name,value)
    verify(partial)
    partial.rename(destination)


def video_shape(path,endpoints=None,fps=None):
    count=0;shape=None
    if fps is not None and (not np.isfinite(fps) or fps<=0):
        raise ValueError('Invalid expected frame rate')
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        stream.codec_context.thread_count=1
        for decoded in container.decode(stream):
            if decoded.is_corrupt:raise ValueError('Corrupt decoded frame')
            current=(decoded.height,decoded.width,3)
            if shape is not None and current!=shape:raise ValueError('Variable frame dimensions')
            shape=current
            # Matching total duration and average FPS can hide duplicate PTS.
            if fps is not None:
                timestamp=None if decoded.pts is None else float(decoded.pts*decoded.time_base)
                if timestamp is None or abs(timestamp-count/fps)>1e-6:
                    raise ValueError(f'Non-CFR timestamp at frame {count}: {timestamp}; expected {count/fps}: {path}')
            count+=1
            if endpoints is not None:
                frame=decoded.to_ndarray(format='bgr24')
                if count==1:endpoints.append(frame.copy())
                if len(endpoints)==1:endpoints.append(frame.copy())
                else:endpoints[1]=frame.copy()
    if not count:raise ValueError('Empty video')
    return (count,*shape)


def sample_files(root):
    return {p.relative_to(root).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
            for p in root.rglob('*') if p.is_file()}


def validate(root,media_endpoints=None):
    sample=json.loads((root/'sample.json').read_text(encoding='utf-8'))
    if not sample['training_ready']:raise ValueError('Sample not approved')
    if (root/sample['conditions']['prompt']).read_text(encoding='utf-8')!=reference_prompt(sample['object_references']):
        raise ValueError('Condition prompt contains unapproved annotation text')
    rgb=video_shape(root/sample['supervision']['target_rgb'],None if media_endpoints is None else media_endpoints.setdefault('rgb_video',[]),fps=sample['fps'])
    proxy=video_shape(root/sample['conditions']['target_whitebox'],None if media_endpoints is None else media_endpoints.setdefault('proxy_video',[]),fps=sample['fps'])
    if rgb!=proxy or rgb[0]!=sample['frame_count']:raise ValueError('Bad paired media shape')
    timings=[]
    for name in (sample['supervision']['target_rgb'],sample['conditions']['target_whitebox']):
        result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=duration,avg_frame_rate','-of','json',str(root/name)],capture_output=True,text=True,check=True)
        stream=json.loads(result.stdout)['streams'][0]
        a,b=map(int,stream['avg_frame_rate'].split('/'))
        timings.append((float(stream['duration']),a/b))
    if any(abs(duration-sample['frame_count']/sample['fps'])>.01 or abs(fps-sample['fps'])>.01 for duration,fps in timings):
        raise ValueError('Paired media timeline mismatch')
    frame_rows=json.loads((root/sample['frame_state_index']).read_text())
    if len(frame_rows)!=sample['frame_count']+1:raise ValueError('Missing frame states')
    times=[f['server_time_estimate_ns'] for f in frame_rows]
    if any(a>=b for a,b in zip(times,times[1:])):raise ValueError('Temporal leakage/discontinuity')
    cameras=json.loads((root/sample['camera_records']).read_text())
    worlds=json.loads((root/sample['world_records']).read_text())
    for frame in frame_rows:
        for field in ('lower_client_sample_seq','upper_client_sample_seq'):
            camera=cameras[str(frame['camera_interpolation'][field])]
            if camera['episode_id']!=sample['episode_id'] or camera['server_clock_domain']!=frame['server_clock_domain']:
                raise ValueError('Wrong camera state')
            if camera['agent_id']!=sample['agent_id'] or camera['client_clock_domain']!=frame['client_clock_domain'] or camera['client_sample_seq']!=frame['camera_interpolation'][field]:
                raise ValueError('Camera composite key mismatch')
        world=worlds[str(frame['world_capture_tick'])]
        if world['episode_id']!=sample['episode_id']:raise ValueError('Wrong world state')
        if world['sample_seq']!=frame['world_capture_tick'] or world['server_clock_domain']!=frame['server_clock_domain']:
            raise ValueError('World composite key mismatch')
    masks=cv2.imread(str(root/sample['supervision']['valid_pixels']),0)
    if masks is None or masks.shape!=rgb[1:3] or not (masks>0).any():raise ValueError('Bad loss mask')
    if not sample['object_references']:raise ValueError('No object memory')
    for edge in sample['object_references']:
        if edge['object_id']!=sample['episode_id']+':entity:'+edge['dataset_entity_id']:
            raise ValueError('Wrong object identity namespace')
        if str(edge['lifecycle']['dataset_entity_id'])!=edge['dataset_entity_id'] or edge['lifecycle']['episode_id']!=sample['episode_id']:
            raise ValueError('Wrong object lifecycle')
        if edge['observed_at_ns']!=times[0] or edge['used_at_ns']!=times[1] or edge['used_until_ns']!=times[-1]:raise ValueError('Reference edge clock mismatch')
        if len(edge['target_projected_bboxes'])!=sample['frame_count']:raise ValueError('Missing object projections')
        for field in ('reference_rgb','reference_whitebox','reference_crop','reference_mask','embedding'):
            if not safe_path(root,edge[field]).is_file():raise ValueError('Missing edge asset')
        mask=cv2.imread(str(root/edge['reference_mask']),0)
        if mask is None or mask.shape!=rgb[1:3] or not mask.any():raise ValueError('Empty object mask')
        if (mask[masks==0]>0).any():raise ValueError('Reference mask includes excluded HUD pixels')
        embedding=np.load(root/edge['embedding'])
        if embedding.shape!=(384,) or not np.isfinite(embedding).all():raise ValueError('Invalid embedding')
    review=json.loads((root/sample['review']).read_text())
    if review['status']!='REVIEWED' or not review['pair_acceptable']:raise ValueError('Missing positive review')
    return sample


def validate_release_sample(path,release):
    media_endpoints={}
    sample=validate(path.parent,media_endpoints)
    source_review=json.loads((release/'visual_review'/sample['sample_id']/'decision.json').read_text())
    embedded_review=json.loads((path.parent/sample['review']).read_text())
    if source_review!=embedded_review:raise ValueError('Review changed after export; re-export required')
    selected=json.loads((release/'visual_review'/sample['sample_id']/'selected_objects.json').read_text())
    approved={o['label'] for o in source_review['objects'] if o['reference_acceptable']}
    candidates={o['entity_id']:o for o in selected if o['label'] in approved}
    if len({e['object_id'] for e in sample['object_references']})!=len(sample['object_references']):
        raise ValueError('Duplicate reference identity')
    for edge in sample['object_references']:
        expected=candidates.get(edge['dataset_entity_id'])
        if expected is None:raise ValueError('Object has no approved visual identity reference')
        if edge['category']!=expected['category'] or edge['lifecycle']!=expected['lifecycle'] or edge['target_projected_bboxes']!=expected['target_bboxes']:
            raise ValueError('Object association disagrees with qualified source evidence')
    from memory.gta_release_media import cached_frame
    segment=sample['episode_id']+'_'+sample['agent_id']+'_'+sample['source_segment']
    frame_rows=json.loads((path.parent/sample['frame_state_index']).read_text())
    reference_index=frame_rows[0]['proxy_frame_index']
    for kind,file in [('rgb_video','rgb.png'),('proxy_video','whitebox.png')]:
        expected=cached_frame(release/'review_frames'/segment,kind,reference_index)
        actual=cv2.imread(str(path.parent/'references'/file))
        if actual is None or actual.shape!=expected.shape or not np.array_equal(actual,expected):
            raise ValueError('Reference does not match exact decoder frame')
        for actual,row in zip(media_endpoints[kind],(frame_rows[1],frame_rows[-1])):
            expected=cached_frame(release/'review_frames'/segment,kind,row['proxy_frame_index'])
            if actual.shape!=expected.shape or cv2.PSNR(actual,expected)<26:
                raise ValueError('Target clip endpoint disagrees with exact source frame')
    return sample


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args()
    root=args.release
    for name in ('qualification_summary.json','visual_review_complete.json','export_complete.json'):
        if not (root/name).is_file():raise ValueError('Processing stage incomplete: '+name)
    for path in (root/'visual_review').glob('*/decision.json'):
        if json.loads(path.read_text()).get('status')!='REVIEWED':
            raise ValueError('Unfinished visual review: '+path.parent.name)
    dataset=root/'dataset';(dataset/'splits').mkdir(exist_ok=True)
    accepted=[];failed=[]
    for path in sorted((dataset/'samples').glob('*/sample.json')):
        try:
            marker=root/'sample_validation'/(path.parent.name+'.json')
            prior=json.loads(marker.read_text()) if marker.exists() else {}
            review_path=root/'visual_review'/path.parent.name/'decision.json'
            review_stat=[review_path.stat().st_size,review_path.stat().st_mtime_ns]
            if prior.get('validator_version')==VALIDATOR_VERSION and prior.get('status')=='PASSED' and prior.get('files')==sample_files(path.parent) and prior.get('review_stat')==review_stat:
                sample=json.loads(path.read_text())
            else:sample=validate_release_sample(path,root)
            accepted.append({'sample_id':sample['sample_id'],'episode_id':sample['episode_id'],
                             'sample_root':path.parent.relative_to(dataset).as_posix(),
                             'objects':len(sample['object_references'])})
        except Exception as error:
            failed.append({'sample':str(path),'error':str(error)})
        if (len(accepted)+len(failed))%50==0:print('validated',len(accepted),'failed',len(failed),flush=True)
    if failed:
        (root/'final_validation_failures.json').write_text(json.dumps(failed,indent=2),encoding='utf-8')
        raise ValueError('Release validation failed; fix before publishing')
    if not accepted:raise ValueError('No accepted samples; cannot publish training release')
    episodes=sorted({r['episode_id'] for r in accepted})
    # Never split alternate views or overlapping windows of one episode.
    split_map={episode:'train' for episode in episodes}
    if len(episodes)>=3:
        split_map[episodes[-2]]='validation';split_map[episodes[-1]]='test'
    splits=defaultdict(list)
    for row in accepted:splits[split_map[row['episode_id']]].append(row)
    for split in ('train','validation','test'):
        (dataset/'splits'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in splits[split]),encoding='utf-8')
    with (dataset/'object_reference_uses.jsonl').open('w',encoding='utf-8') as uses:
        for row in accepted:
            sample=json.loads((dataset/row['sample_root']/'sample.json').read_text(encoding='utf-8'))
            for edge in sample['object_references']:
                uses.write(json.dumps({'sample_root':row['sample_root'],'split':split_map[row['episode_id']],
                                       'frame_state_index':sample['frame_state_index'],**edge})+'\n')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'memory/gta_training_dataset.py',dataset/'dataset.py')
    ledger=[json.loads(s) for s in (root/'source_ledger.jsonl').read_text().splitlines()]
    processed={r['source']['source_manifest']:r for r in json.loads((root/'qualification_summary.json').read_text())}
    for row in ledger:
        if row['status']=='PENDING':
            summary=processed.get(row['source_manifest'])
            if summary is None:raise ValueError('Source segment never processed')
            row['status']=summary['status']
            row['decoded_frames']=summary['decoded_frames']
            row['window_counts']=summary.get('windows',{})
    decisions=[]
    accepted_ids={r['sample_id'] for r in accepted}
    for path in (root/'qualification').glob('*/windows.jsonl'):
        for window in map(json.loads,path.read_text().splitlines()):
            verdict=root/'export_verdicts'/(window['sample_id']+'.json')
            status='ACCEPTED' if window['sample_id'] in accepted_ids else (json.loads(verdict.read_text())['status'] if verdict.exists() else window['status'])
            if status=='NEEDS_VISUAL_REVIEW':raise ValueError('Unfinished candidate '+window['sample_id'])
            if status=='EXCLUDED_REVIEW_UNAVAILABLE':raise ValueError('Review unavailable for candidate '+window['sample_id'])
            reasons=window.get('reasons',[])
            if verdict.exists() and status!='ACCEPTED':
                reason=json.loads(verdict.read_text()).get('reason')
                if reason:reasons=[*reasons,reason]
            row={'sample_id':window['sample_id'],'status':status,'reasons':reasons}
            review_dir=root/'visual_review'/window['sample_id']
            if (review_dir/'decision.json').exists():
                visual=json.loads((review_dir/'decision.json').read_text())
                row['visual_review_status']=visual['status']
                row['visual_review_reason']=visual.get('reason',visual.get('error'))
                row['object_review_decisions']=visual.get('objects',[])
                if (review_dir/'selected_objects.json').exists():
                    row['review_label_to_entity']={o['label']:o['entity_id'] for o in json.loads((review_dir/'selected_objects.json').read_text())}
            decisions.append(row)
    for split in splits:
        loaded=GTAObjectMemoryDataset(dataset,split)[0]
        if 'target_rgb' in loaded['conditions']:raise ValueError('Supervision leaked into conditions')
    report={'status':'RELEASE_VALIDATED','source_segments':len(ledger),
            'source_status_counts':dict(Counter(r['status'] for r in ledger)),
            'window_status_counts':dict(Counter(r['status'] for r in decisions)),
            'accepted_samples':len(accepted),'object_reference_edges':sum(r['objects'] for r in accepted),
            'target_frames':120*len(accepted),'split_samples':{k:len(v) for k,v in splits.items()},
            'episode_split_map':split_map,
            'quality_scope':'Every accepted video fully decoded; exact historical reference pixels and target endpoint alignment checked; each frame actual camera/world joined; 8 target frames per window visually reviewed; reference box SAM2 mask and DINOv2 auxiliary embedding validated.',
            'limitations':['Not every intermediate RGB frame manually reviewed','Source clock alignment is estimated',
                           'At most four largest eligible reference objects per clip, minimum projected size 24 pixels; not exhaustive all-object visibility',
                           'Historical reference gap is 120 source frames; this release does not test long-term revisit memory',
                           'Coarse proxy poses/vehicle wireframes are intentional','Static instance memory excluded: collision geometry and verified static identity unavailable',
                           'SAM masks are model estimates, not manual ground truth','Episode split prevents adjacent-view leakage, not shared GTA-map scenery across episodes']}
    (dataset/'acceptance_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (dataset/'source_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger),encoding='utf-8')
    (dataset/'window_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in decisions),encoding='utf-8')
    readme='''# GTA Object-Centric Memory Training Data\n\nThis release is approved for coarse whitebox-conditioned RGB learning with historical object appearance references. It is not exact articulated-pose supervision or pixel-perfect geometry.\n\nEach sample contains target RGB supervision, a target whitebox video, a strictly earlier RGB/whitebox reference pair, object reference-use edges, SAM2 masks, DINOv2 embeddings, prompt, per-frame metadata, actual camera and world records, and visual QA evidence. Read acceptance_report.json for measured counts and exclusions.\n\nUse dataset.py: GTAObjectMemoryDataset(root, split). It returns conditions separately from supervision. Apply supervision.valid_pixels to the training loss; otherwise HUD pixels will contaminate training. Reference RGB HUD regions are zeroed by the loader, not hallucinated/inpainted. Object masks are estimated, optional inputs.\n\nTarget clips have 120 frames at the recorded FPS, normally 24 FPS. This is a model-independent dataset, not a claim that an unmodified H3 trainer accepts 120 frames. A trainer requiring 4n+1 frames must use a documented temporal adapter and keep paired streams aligned.\n\nHistorical exterior appearance and target dynamic state are distinct. Hidden properties are unknown; do not copy historical pose/door state into the target. Every use edge identifies the observed timestamp, target use interval, entity lifecycle, and appearance observation revision.\n\nNo static building/ground instance-memory labels are supplied without reliable geometry/identity; these appear as unreferenced scene context. No validation/test overlap by episode or alternate camera view. Shared GTA map scenery may exist across episodes. See source_ledger.jsonl and window_ledger.jsonl for complete processing accountability.\n'''
    (dataset/'README.md').write_text(readme,encoding='utf-8')
    (dataset/'README_zh.md').write_text('''# GTA Object-Centric Memory 训练数据

这是粗白模条件下的物体历史外观记忆训练数据。准确规模、数据划分和排除原因以 acceptance_report.json、source_ledger.jsonl、window_ledger.jsonl 为准。

## 使用

把 release_manifest.zip 和全部 episode ZIP 解压到同一个目录，然后安装 requirements.txt。

```python
from dataset import GTAObjectMemoryDataset
data = GTAObjectMemoryDataset("解压目录", "train")
sample = data[0]
conditions = sample["conditions"]
supervision = sample["supervision"]
```

conditions 包含目标白模视频、历史 RGB/白模参考帧、物体参考关系提示词，以及可直接使用的 mask/crop。supervision 包含目标 RGB 和 valid_pixels；训练 loss 必须应用这个像素掩码，避免学习 HUD。

```bash
python verify_delivery.py --dataset 解压目录
```

默认验证全部样本元数据和引用账本，并真实读取每个非空 split 的一个样本。发布阶段已对全部导出视频重新解码并检查首尾帧对应。

## 不应输入模型的材料

visual_review.json 和 qa 图片看过目标 RGB，仅供审核。完整 lifecycle 含未来消失信息，仅供追溯。不要把 sample.json/object_reference_uses.jsonl 整体直接作为条件；请使用 dataset.py 的字段白名单。DINO 特征在 auxiliary_object_features 中，不属于默认训练条件。

## 数据边界

每段120帧，通常24 FPS，参考帧比目标起点早120个源帧。目标视频采用相同帧索引的恒定播放节奏，原始采集时间保留在 frame_state_index.json。首帧、世界状态与相机记录均保留对应关系。

每个窗口视觉审核参考帧及8张目标帧，不声称人工逐帧检查；SAM mask 也是模型估计。每段最多4个较大、持续可投影的物体。建筑和地面没有可靠实例级记忆标签，作为背景上下文保留。当前不覆盖长期离开再返回的记忆测试。

按整个 episode 划分 train/validation/test，同一 episode 的不同机位不会跨 split；不同 episode 仍可能共享 GTA 地图。120帧是模型无关格式，H3 trainer 若要求4n+1帧，需要对RGB、白模、投影和状态同时做一致的时间选择。本交付不宣称已经完成模型微调。
''',encoding='utf-8')
    code=dataset/'processing_code';code.mkdir(exist_ok=True)
    project=Path(__file__).resolve().parents[1]
    for folder in ('scripts','memory'):
        for source in (project/folder).glob('*gta*.py'):
            destination=code/folder/source.name;destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source,destination)
    shutil.copyfile(project/'requirements-data.txt',code/'requirements-data.txt')
    for source in (project/'docs').glob('gta_release*.md'):
        shutil.copyfile(source,code/source.name)
    shutil.copyfile(project/'scripts/verify_gta_delivery.py',dataset/'verify_delivery.py')
    (dataset/'requirements.txt').write_text('numpy>=1.24\nopencv-python>=4.8\n',encoding='utf-8')
    for name in ('conditioning_repairs.json','timing_repairs.json','random_seek_retirement.json'):
        if (root/name).exists():shutil.copyfile(root/name,dataset/name)
    verification=subprocess.run([__import__('sys').executable,str(dataset/'verify_delivery.py'),'--dataset',str(dataset)],capture_output=True,text=True,encoding='utf-8')
    (dataset/'delivery_verification.json').write_text(verification.stdout,encoding='utf-8')
    if verification.returncode:raise ValueError('Standalone delivery verification failed: '+verification.stderr[-1000:])
    archives=[]
    for episode in episodes:
        archive_path=root/(episode+'_training.zip')
        files=[(path,path.relative_to(dataset).as_posix()) for row in accepted if row['episode_id']==episode
               for path in sorted((dataset/row['sample_root']).rglob('*')) if path.is_file()]
        write_archive(archive_path,files)
        archives.append({'file':archive_path.name,'bytes':archive_path.stat().st_size})
    files=[(path,path.relative_to(dataset).as_posix()) for path in sorted(dataset.rglob('*'))
           if path.is_file() and 'samples' not in path.relative_to(dataset).parts]
    write_archive(root/'release_manifest.zip',files,{'archives.json':json.dumps(archives,indent=2)})
    (root/'DELIVERY.json').write_text(json.dumps({'report':report,'archives':archives},indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
