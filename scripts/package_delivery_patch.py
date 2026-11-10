"""Package the small, reproducible patch for handoff after real repair verification."""
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

SOURCE=Path('D:/GTA_memory_patch')
DEST=Path('C:/Users/zjc/Desktop/GTA_memory_patch')

def main():
    spec=json.loads((SOURCE/'patch.json').read_text(encoding='utf-8'))
    check=json.loads((SOURCE/'integration_verification.json').read_text())
    assert check['all_manifest_filters_exact'] and check['all_original_video_backups_identical']
    DEST.mkdir(exist_ok=True)
    for p in SOURCE.rglob('*'):
        if p.is_file() and p.suffix!='.mp4' and p.name!='progress.json':
            dest=DEST/p.relative_to(SOURCE)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
    shutil.copy2(Path(__file__).with_name('apply_delivery_patch.py'),DEST/'apply_patch.py')
    shutil.copy2(Path(__file__).with_name('verify_reaudit_pts_repair.py'),DEST/'repair_video.py')
    report=spec['summary']
    issues=[json.loads(s) for s in (SOURCE/'issues.jsonl').read_text(encoding='utf-8').splitlines()]
    reasons=Counter(s for r in issues for s in r['reasons'])
    doc=f'''# 发给训练同学的 GTA memory 修正补丁

## 一句话说明

这是后一次完整交付包（7085条，原ZIP大小{spec['source_archive_bytes']:,}字节）的修正补丁。
177条只修视频时间戳；329条暂时移出训练索引；应用后6756条。不是重新下载整个数据集。

## 直接运行

先解压本补丁，进入这个目录；将路径换成已经解压的数据集根目录（里面有samples、manifests、dataset.py）。

```bash
python -m pip install av numpy
python apply_patch.py --root /path/to/GTA_memory_training --check --rebuild-media
python apply_patch.py --root /path/to/GTA_memory_training --apply --rebuild-media
```

Windows路径例子：`--root "E:\\GTA_memory_training"`。`--check`检查补丁适用性，不修改数据；`--apply`才会执行。

小包不携带约数GB的无损视频：脚本会利用师姐已有的原视频在本地重建177条、共354个视频，验证每帧解码RGB像素与原件完全相同、每段120帧且时间戳严格24fps，然后备份并替换。需要CPU，不需要GPU或模型API。补丁目录和数据盘请各留至少10GB空间。过程中出现报错会停止，勿在报错后手动覆盖索引。

原视频和索引备份在数据根目录`patch_backup/`，已完成的重建保存在补丁目录`payload/`，中断可重新执行同一命令。运行时停止正在读取这些文件的训练任务。

## 为什么改

1. 177条存在重复/不均匀视频时间戳，但帧内容可以保留：按原来的解码顺序重建恒定帧率。两路视频一起处理，不抽帧、不补帧、不改图像。
2. 另外321条涉及采集间隔、源时钟比例或已发现的对齐问题：不能靠重新编码修好，暂时排除。其中3条同时有时间戳问题，也排除，不重复计算。
3. 再追查出8条的历史参考来自异常时钟源：目标本身正常也不能保证历史条件正确，同样排除。共329条。

## 修正后数量

|划分|数量|
|---|---:|
'''
    for key,n in report['splits'].items():doc+=f'|{key}|{n}|\n'
    doc+='\n|任务|数量|\n|---|---:|\n'
    for key,n in report['tasks'].items():doc+=f'|{key}|{n}|\n'
    doc+='''
主训练索引、validation/test和全部10轮train_round索引均已同步过滤，保留原顺序，不重新划分split。类别比例有小幅变化，详细计数在payload/DATASET_REPORT.json。不要绕过索引直接扫描samples文件夹，被排除的原文件并没有删除。自定义dataloader可读excluded_sample_ids.txt过滤。

## 包内文件

- issues.csv / issues.jsonl：逐样本原因和REPAIR_CFR / EXCLUDE操作，含原split和task。
- excluded_sample_ids.txt：329条排除名单。
- payload/manifests/：全部修正后的索引。
- payload/DATASET_REPORT.json：修正后计数。
- verification/：本机177条、354个真实视频逐帧比较的通过记录。
- integration_verification.json：在原包的真实文件副本上应用补丁、重复应用及备份验证。
- lightweight_verification.json：不携带视频的小包，在真实原视频上现场重建的验证记录。
- visual_spotcheck_advisory.json：额外24条随机视觉抽查记录，9条待核对，不算程序确认错帧，不在本补丁中额外剔除。

## 原因名对照

|字段|含义|
|---|---|
|encoded_media_timing_or_decode_issue|本包具体为视频编码时间戳异常，完整解码检查未发现坏帧|
|capture_time_gap_over_two_output_frames|采集时间间隔超过两个输出帧|
|server_rgb_pts_rate_outside_2_percent|样本时间跨度内世界时钟与RGB时间比例异常|
|source_clock_rate_flagged|源片段时钟比例异常|
|reference_uses_flagged_clock_source|历史参考取自上述异常源片段|
|REFERENCE_SPATIAL_MISMATCH / REFERENCE_CAMERA_MISMATCH|人工发现历史参考空间/相机对应异常|
|CONFIRMED_*|人工复核已发现的目标对齐/交互问题|

以上原因会重叠，不能将各原因数量直接相加。

## 本次范围

只针对后一次7085条完整包；不适用于前一次旧包。源RGB/世界时钟重新标定、白模几何重建不在本补丁中。此补丁解决已定位的异常，不代表6756条已逐帧人工验收。不要把暂时剔除写成“对齐已修复”。
'''
    (DEST/'README.md').write_text(doc,encoding='utf-8')
    (DEST/'requirements.txt').write_text('av\nnumpy\n')
    archive=DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,'GTA_memory_patch/'+p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(n.endswith('.mp4') for n in z.namelist())
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'records':spec['remaining'],'reason_counts':reasons},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
