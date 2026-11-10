"""Package historical video evidence for a local report and manual Feishu upload."""
import csv
import html
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from server_exec import connect

REPORT=ROOT/'reports/feishu_20261013'
SELECTED={
    'V01_early_11s':'action_wukong_outline_20261008/first_action_11s.mp4',
    'V02_memory_hold':'appearance_memory_20261012/generated.mp4',
    'V03_small_step':'refined_rollouts_20261012/segment_00.mp4',
    'V04_posture_sequence':'posture_live_20261010/accepted_sequence.mp4',
    'V05_continuous_attempt':'continuous_motion_20261012/experiment_sequence_labeled.mp4',
    'V06_follow_camera':'natural_walk_dolly_20261012/generated.mp4',
    'V07_orbit_baseline':'identity_repair_20261013/prompt_regression_run1/baseline/generated.mp4',
    'V08_prompt_comparison':'identity_repair_20261013/prompt_regression_run1/detail_comparison.mp4',
    'V09_architecture_comparison':'identity_repair_20261013/prompt_regression_run1/architecture_comparison.mp4',
    'V10_early_whitebox':'action_wukong_outline_20261008/whitebox_export/first_action_11s_gray.mp4',
    'V11_actual_early_control':'action_wukong_outline_20261008/whitebox_export/first_action_11s_layout.mp4',
}


def kind(path):
    name=Path(path).name
    if 'cwm_' in path:return '外部参考 非本系统生成'
    if any(x in name for x in ['comparison','sequence','accepted_action','first_action','retreat','prefix','half_speed']):
        return '拼接或导出 名称初筛'
    if any(x in name for x in ['source','control_input','control_canny','depth','whitebox','proxy']):
        return '输入或白模 名称初筛'
    return '生成候选或其他 待逐项核验'


def main():
    remote=json.loads((REPORT/'server_videos.json').read_text())
    local=json.loads((REPORT/'local_videos.json').read_text())
    client=connect()
    try:
        with client.open_sftp() as sftp:
            for i,row in enumerate(remote):
                target=REPORT/'videos/server'/row['relative_path']
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or target.stat().st_size!=row['bytes']:
                    sftp.get(row['path'],str(target))
                if (i+1)%50==0:print('Downloaded',i+1,'/',len(remote),flush=True)
    finally:client.close()
    attachment=REPORT/'attachments';attachment.mkdir(exist_ok=True)
    for name,relative in SELECTED.items():
        shutil.copy2(ROOT/'outputs'/relative,attachment/(name+'.mp4'))
    rows=[]
    for location,inventory in [('server',remote),('local',local)]:
        for row in inventory:
            if location=='server':
                target=REPORT/'videos/server'/row['relative_path']
            elif row['readable_header']:
                target=REPORT/'videos/local'/row['relative_path']
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(Path(row['path']),target)
            else:target=None
            rows.append({'location':location,'path':row['path'],'relative_path':row['relative_path'],
                'category_hint':kind(row['relative_path']),'duration_seconds':row['duration_seconds'],
                'resolution':f"{row['width']}x{row['height']}",'readable_header':row['readable_header'],
                'bundle_path':target.relative_to(REPORT).as_posix() if target else '',
                'review_status':'未逐条严格外观验收'})
    with (REPORT/'video_inventory.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    md=['# 历史视频全量索引','',
        '库存快照包含生成候选、失败重试、白模控制、拼接和外部参考。重复文件未去重，条目数不是独立生成次数。分类为文件名初筛，不能据此判断 ACCEPT 或生成类型。','',
        '本索引中的相对链接在本地文件包中使用；飞书需上传相应视频并替换为附件链接。','']
    body=['<!doctype html><meta charset="utf-8"><title>历史视频索引</title>',
          '<style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#222}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}table{width:100%;table-layout:fixed}a{color:#1264aa}</style>',
          '<h1>历史视频索引</h1><p>文件库存，不是生成次数。分类未经逐条确认。点击文件在浏览器播放。未对全部视频作严格外观验收。</p>']
    for location in ['server','local']:
        md.extend([f'## {location} 文件',''])
        body.append(f'<h2>{location}</h2><table><tr><th>文件</th><th>秒数与分辨率</th><th>初筛类别</th></tr>')
        for row in rows:
            if row['location']!=location:continue
            path=row['bundle_path'];label=row['relative_path'];duration=row['duration_seconds']
            info=f"{duration:.2f}s" if duration is not None else '无法读取'
            md.append(f"- [{label}]({quote(path)}) · {info} · {row['category_hint']}" if path else f'- {label} · 本地副本损坏 未打包')
            link=f'<a href="{quote(path)}">{html.escape(label)}</a>' if path else html.escape(label)+' 本地副本损坏'
            body.append(f'<tr><td>{link}</td><td>{info} {row["resolution"]}</td><td>{row["category_hint"]}</td></tr>')
        body.append('</table>')
    (REPORT/'video_index.md').write_text('\n'.join(md),encoding='utf-8')
    (REPORT/'video_index.html').write_text('\n'.join(body),encoding='utf-8')
    print('Packaged',len(remote),'server artifacts and',sum(r['readable_header'] for r in local),'local artifacts')


if __name__=='__main__':main()
