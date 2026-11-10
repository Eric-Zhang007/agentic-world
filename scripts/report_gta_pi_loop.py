"""Build local audit and playable evidence from completed Pi trial artifacts."""
import html
import json
from pathlib import Path
import sqlite3
from contextlib import closing

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gta_pi_loop_20261102'
REMOTE='/home/star/agentic_world/outputs/gta_pi_loop_20261102'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def local(path):
    return OUT/str(path).removeprefix(REMOTE+'/')


def audit(name):
    base=OUT/name
    initial_name=name.split('/')[0]
    with closing(sqlite3.connect((base/'world.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
        meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM world_meta')}
    history=meta.get('whitebox_history',[])
    scene=read(OUT/initial_name/'scene.json')
    static={e['id']:e for e in scene['entities'] if e['kind'] not in ['actor','staff']}
    checks=[];previous_scene=scene;previous_frame=str(REMOTE+'/'+initial_name+'/input.png')
    for index,record in enumerate(history):
        artifacts=record['artifacts'];plan=local(artifacts['planned_timeline']).parent
        after=read(local(artifacts['corrected_timeline']).parent/'world_after.json')
        receipt=read(local(artifacts['corrected_timeline']).parent/'receipt.json')
        render=local(artifacts['video']).parent
        command=read(render/'generation_metadata.json')['command']
        first=command[command.index('--reference-image')+1]
        checks.append({'segment':index+1,'revision':record['revision_after'],
            'starts_from_previous_actual_scene':read(plan/'scene.json')==previous_scene,
            'actual_last_frame_used_as_next_first_frame':first==previous_frame,
            'static_entities_unchanged':{e['id']:e for e in after['entities'] if e['id'] in static}==static,
            'revision_contiguous':record['revision_after']==index+1,
            'saved_commit_matches_history':receipt==record,
            'action_complete':record['action_complete'],
            'goal':record['action'].get('goal'),
            'continuation_goal':record['visual_review'].get('continuation_goal'),
            'root_correction_count':len(record['visual_review'].get('corrections',[])),
            'motion_assessments':record['visual_review'].get('motion_assessments',[]),
            'camera_corrected':bool(record['visual_review'].get('camera_correction')),
            'video':str(local(artifacts['video']))})
        previous_scene=after;previous_frame=artifacts['last_frame']
    summaries=[{'path':str(p.relative_to(base)),'data':read(p)} for p in base.glob('*summary.json')]
    inherited_history_unchanged=None
    if name!=initial_name:
        source_name=read(base/'launch.json')['source'] if (base/'launch.json').exists() else initial_name
        with closing(sqlite3.connect((OUT/source_name/'world.sqlite').as_uri()+'?mode=ro',uri=True)) as source:
            inherited=json.loads(source.execute("SELECT value FROM world_meta WHERE key='whitebox_history'").fetchone()[0])
        inherited_history_unchanged=history[:len(inherited)]==inherited
    return {'case':name,'revision':meta['whitebox_revision'],'accepted_segments':len(history),
            'completed_action_reviews':sum(bool(r['action_complete']) for r in history),
            'inherited_history_unchanged':inherited_history_unchanged,
            'world_time':meta.get('tick'),'pending_action':meta.get('whitebox_video_action'),
            'chain_checks':checks,'summaries':summaries}


def main():
    names=['rural','urban','harbor']
    for branch in ['rural/continuation','rural/road_edge','urban/npc_walk']:
        if (OUT/branch/'world.sqlite').exists():names.append(branch)
    audits=[audit(name) for name in names]
    (OUT/'closed_loop_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    page=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GTA Pi 闭环测试</title>
<style>body{font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1500px;color:#222;background:#fff}h1{font-size:26px}h2{font-size:21px;margin-top:32px}img{max-width:100%;height:auto}video{width:100%;background:#111;aspect-ratio:16/9}section{border-top:1px solid #ccc;padding:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}a{color:#176348}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>GTA 单图建模与 Pi 闭环实测</h1><p>左：RGB；中：数据集白模（Agent 未读取）；右：Agent 单图重建。</p>
<img src="reconstruction_comparison.jpg" alt="三组单图重建对照">''']
    for item in audits:
        name=item['case'];base=OUT/name
        page.append(f'<h2>{html.escape(name)}：接受 {item["accepted_segments"]} 段，动作完成判定 {item["completed_action_reviews"]} 次，世界时间 {item["world_time"]:.3f}s</h2>')
        page.append('<p>段数包含未完成动作的续接；完成判定来自 Agent，需结合视频复核，不等于独立测量。</p>')
        if name=='rural/continuation':
            page.append('<p>独立分支继承乡镇的前三段历史；新增部分使用图片工具改进版，不计入原基线。</p>')
        for filename,label in [('accepted_rollout.mp4','连续已接受 RGB'),('accepted_comparison.mp4','连续白模 / RGB 对照')]:
            if (base/filename).exists():
                page.append(f'<p><a href="{name}/{filename}">{label}</a></p>')
        if (base/'launch.json').exists():
            launch=read(base/'launch.json')
            page.append(f'<p>独立续接分支，继承自 {html.escape(launch["source"])}；历史段数包含继承部分。</p>')
        if (base/'trial_summary.json').exists():
            page.append('<details><summary>本次试验结果与未完成目标</summary><pre>'+html.escape(json.dumps(read(base/'trial_summary.json'),ensure_ascii=False,indent=2))+'</pre></details>')
        if (base/'boundary_frames.jpg').exists():
            page.append(f'<details><summary>段落边界：上段末两帧 / 下段首两帧</summary><img src="{name}/boundary_frames.jpg" alt="实际解码的段落边界对照"></details>')
        for video in sorted(base.rglob('controlled.mp4')):
            if video.parent.name!='render':continue
            if any(other!=name and other.startswith(name+'/') and video.is_relative_to(OUT/other) for other in names):continue
            attempt=video.parent.parent;whitebox=attempt/'compiled/whitebox.mp4'
            page.append('<section><p>'+html.escape(str(attempt.relative_to(base)))+'</p><div class="pair">')
            for label,path in [('计划白模',whitebox),('H3 实际生成',video)]:
                if path.exists():
                    page.append(f'<div><p>{label}</p><video controls preload="metadata" src="{html.escape(path.relative_to(OUT).as_posix())}"></video></div>')
            page.append('</div>')
            for detail in ['player_motion.jpg','npc_motion.jpg']:
                if (attempt/detail).exists():
                    page.append(f'<details><summary>密集动作帧：{detail}</summary><img src="{(attempt/detail).relative_to(OUT).as_posix()}" alt="每 0.125 秒真实视频局部帧，无插帧"></details>')
            for review in attempt.rglob('review.json'):
                page.append('<details><summary>Agent 审查</summary><pre>'+html.escape(json.dumps(read(review),ensure_ascii=False,indent=2))+'</pre></details>')
            page.append('</section>')
    page.append('<p><a href="closed_loop_audit.json">状态接续核验</a> · <a href="asset_audit.json">资源与数据集资产核对</a></p></html>')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print(json.dumps([{k:x[k] for k in ['case','accepted_segments','revision','world_time']} for x in audits]))


if __name__=='__main__':main()
