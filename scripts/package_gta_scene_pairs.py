"""Verify scene-frame assets, keep episode splits, and package a portable dataset."""
import argparse
from collections import Counter
import html
from pathlib import Path
import shutil
import zipfile

import cv2
import orjson


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('D:/GTA_scene_pairs_20261019'))
    parser.add_argument('--delivery',type=Path,default=Path('E:/GTA_scene_pairs_20261019'))
    args=parser.parse_args()
    root=args.root;dataset=root/'dataset'
    report=orjson.loads((root/'report.json').read_bytes())
    source_report=orjson.loads(Path('D:/GTA_memory_release_20261018/DELIVERY.json').read_bytes())['report']
    splits=source_report['episode_split_map']
    rows=[orjson.loads(line) for line in (dataset/'samples.jsonl').read_bytes().splitlines()]
    split_rows={name:[] for name in ('train','validation','test')}
    for row in rows:
        folder=dataset/row['sample_root'];sample=orjson.loads((folder/'sample.json').read_bytes())
        states=[orjson.loads((folder/(side+'_state.json')).read_bytes()) for side in ('reference','target')]
        a,b=states
        if not a['time_ns']<b['time_ns'] or a['episode_id']!=b['episode_id'] or a['server_clock_domain']!=b['server_clock_domain']:
            raise ValueError('Invalid reference chronology/world')
        if sample['reference_time_ns']!=a['time_ns'] or sample['target_time_ns']!=b['time_ns']:
            raise ValueError('Reference metadata mismatch')
        for side,state in zip(('reference','target'),states):
            images=[]
            for kind in ('rgb','whitebox'):
                image=cv2.imread(str(folder/(side+'_'+kind+'.png')))
                if image is None or list(image.shape[1::-1])!=state['resolution']:
                    raise ValueError('Missing or invalid image')
                images.append(image)
            if images[0].shape!=images[1].shape:raise ValueError('Unpaired dimensions')
        if sample.get('training_status')=='REVIEWED_SCENE_VIDEO':
            split_rows[splits[row['episode_id']]].append(row)
    (dataset/'splits').mkdir(exist_ok=True)
    for split,items in split_rows.items():
        (dataset/'splits'/(split+'.jsonl')).write_bytes(b''.join(orjson.dumps(r)+b'\n' for r in items))
    report['split_counts']={k:len(v) for k,v in split_rows.items()}
    report['decoded_and_verified_pairs']=len(rows)
    report['training_status_counts']=dict(Counter(orjson.loads((dataset/r['sample_root']/'sample.json').read_bytes()).get('training_status','PAIR_ONLY') for r in rows))
    report['limitations']=['Automatic camera proximity and background feature verification, not exhaustive visual annotation.',
        'Sparse trajectory samples every 120 frames; intersections between samples can be missed.',
        'No object correspondence. Foreground masks are coarse projections, not ground truth.',
        'No claim of unchanged weather, lighting, or unseen surfaces.',
        'Cross-camera intersection is not evidence of a single camera leaving and returning.',
        'Reviewed training samples include 120-frame target video; five frames visually inspected per video. No H3 training performed.']
    gallery=[]
    for row in rows:
        sample=orjson.loads((dataset/row['sample_root']/'sample.json').read_bytes())
        key=html.escape(row['sample_id']);relative=html.escape(row['sample_root'])
        gallery.append(f'<article><h2>{key}</h2><p>{html.escape(sample.get("training_status","PAIR_ONLY"))} | {html.escape(sample["kind"])} | '
            f'{sample["age_sec"]:.1f} s | {sample["camera_distance"]:.2f} world units | '
            f'{sample["view_angle_deg"]:.1f} degrees</p><a href="{relative}/comparison.jpg">'
            f'<img loading="lazy" src="{relative}/comparison.jpg" alt="{key}"></a>'
            f'<p><a href="{relative}/reference_rgb.png">Reference RGB</a> | '
            f'<a href="{relative}/reference_whitebox.png">Reference whitebox</a> | '
            f'<a href="{relative}/target_rgb.png">Target RGB</a> | '
            f'<a href="{relative}/target_whitebox.png">Target whitebox</a> | '
            f'<a href="{relative}/sample.json">Metadata</a></p>'+
            (f'<video controls preload="none" width="640" src="{relative}/target_rgb.mp4"></video>'
             f'<video controls preload="none" width="640" src="{relative}/target_whitebox.mp4"></video>'
             if (dataset/row['sample_root']/'target_rgb.mp4').exists() else '')+'</article>')
    (dataset/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>GTA scene reference pairs</title><style>body{font:16px system-ui;margin:24px;background:#f5f6f7;color:#202020}'
        'main{max-width:1280px;margin:auto}article{padding:20px 0;border-bottom:1px solid #bbb}'
        'img{max-width:100%;height:auto}h2{font-size:18px;overflow-wrap:anywhere}a{color:#176b47}</style>'
        '<main><h1>GTA scene reference pairs</h1><p>Automatic background-overlap verification; '
        'see report.json for coverage and limitations.</p>'+''.join(gallery)+'</main>',encoding='utf-8')
    (dataset/'report.json').write_bytes(orjson.dumps(report,option=orjson.OPT_INDENT_2))
    for name in ('pair_ledger.jsonl','mining_config.json','index_report.json'):
        shutil.copy2(root/name,dataset/name)
    shutil.copy2(Path('docs/gta_scene_pairs.md'),dataset/'README.md')
    code=dataset/'processing_code';code.mkdir(exist_ok=True)
    for name in ('scripts/build_gta_scene_pairs.py','scripts/package_gta_scene_pairs.py','memory/gta_scene_pairs.py','scripts/test_gta_scene_pairs.py'):
        dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    args.delivery.mkdir(parents=True,exist_ok=True)
    target=args.delivery/'scene_pairs.zip'
    if target.exists():raise ValueError('Existing delivered archive is not overwritten')
    partial=args.delivery/'scene_pairs.zip.partial'
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_STORED,allowZip64=True) as archive:
        for path in sorted(dataset.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(dataset).as_posix())
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip():raise ValueError('Archive integrity failure')
    partial.rename(target)
    shutil.copy2(dataset/'README.md',args.delivery/'README.md')
    shutil.copy2(dataset/'report.json',args.delivery/'report.json')
    print(orjson.dumps({'archive':str(target),'bytes':target.stat().st_size,**report},option=orjson.OPT_INDENT_2).decode())


if __name__=='__main__':main()
