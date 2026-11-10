"""Assemble the meeting handout from actual accepted histories and local evidence."""
from pathlib import Path
import html
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/meeting_demo'
DEST=Path.home()/'Desktop/Agentic_World_Demo'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')


def main():
    audits=read(SOURCE/'delivery_audit.json')
    assert all(a['accepted_segments']>=2 for a in audits), 'Two accepted segments per mode required'
    DEST.mkdir(exist_ok=True)
    for name in ('meeting_demo.mp4','demo_keyboard.mp4','demo_action.mp4','keyboard_rgb.mp4',
                 'action_rgb.mp4','pipeline.svg','pipeline.png','closed_loop_evidence.png',
                 'projection_review.png','projection_review.json','delivery_audit.json','input.png','source.json'):
        shutil.copy2(SOURCE/name,DEST/name)
    if (SOURCE/'ui_review').exists():
        shutil.copytree(SOURCE/'ui_review',DEST/'ui_review',dirs_exist_ok=True)
    sections=[];all_calls=[]
    for audit in audits:
        lane=audit['lane'];label='键鼠输入' if lane=='keyboard' else 'Action prompt'
        target=DEST/'evidence'/lane
        target.mkdir(parents=True,exist_ok=True)
        for path in (SOURCE/lane).rglob('*'):
            if not path.is_file():continue
            if path.name.startswith(('console','launcher')) or path.suffix=='.log':continue
            if path.suffix not in ('.json','.txt','.mp4','.jpg','.png','.sqlite'):continue
            relative=path.relative_to(SOURCE/lane)
            (target/relative).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target/relative)
        record=read(SOURCE/lane/'interaction_record.json')
        starts={e['id']:e for e in record if e['event']=='tool_execution_start'}
        calls=[]
        for event in record:
            if event['event']!='tool_execution_end':continue
            start=starts.get(event['id'],{})
            if start.get('tool')!='world':continue
            args=start.get('args',{})
            calls.append({'operation':args.get('operation'),'arguments':args.get('args'),
                          'error':event.get('error',False),'result':event.get('text')})
        all_calls.append({'mode':lane,'world_tool_calls':calls})
        rows=[]
        sections.append(f'## {label}\n')
        sections.append(f"实际接受 {audit['accepted_segments']} 段，世界时间推进到 {audit['world_seconds']:.2f} 秒。\n")
        for check in audit['checks']:
            review=check['review'];n=check['segment']
            sections.append(f"### 第 {n} 段：版本 {check['revision_before']} → {check['revision_after']}\n")
            sections.append('动作状态：'+('本次输入完成。' if check['action_complete'] else '接受本段进展，动作目标仍有剩余。')+'\n')
            sections.append('GPT 实际审查摘录（原文）：\n')
            sections.extend('> '+str(item)+'\n' for item in review.get('evidence',[]))
            sections.append('实际状态处理：\n')
            for actor in review.get('actor_assessments',[]):
                sections.append(f"- `{actor['id']}`：`{actor['status']}`。\n")
            for correction in review.get('corrections',[]):
                sections.append(f"- `{correction['id']}` 的修正：`{json.dumps(correction.get('keyframes',[]),ensure_ascii=False)}`。\n")
            for motion in review.get('motion_assessments',[]):
                sections.append(f"- `{motion['id']}` 末态运动：`{motion['phase']}`，速度估计 `{motion['velocity']}`；不确定性：{motion['uncertainty']}\n")
            if n>1:
                sections.append(f"接续核对：下一段白模初态等于上段提交终态：{check['starts_from_previous_actual_state']}；下一段 RGB 首帧使用上段真实末帧：{check['uses_last_accepted_rgb']}。\n")
            sections.append(f"完整审查和提交回执：`evidence/{lane}/`；可见工具调用顺序见 `真实工具调用.json`。\n")
    write(DEST/'真实工具调用.json',json.dumps(all_calls,ensure_ascii=False,indent=2))
    write(DEST/'真实闭环记录.md','\n'.join(sections))
    interactive=SOURCE/'interactive'
    if interactive.exists():
        for path in interactive.rglob('*'):
            if not path.is_file() or path.suffix not in ('.json','.txt','.mp4','.jpg','.png','.sqlite'):continue
            target=DEST/'evidence/interactive'/path.relative_to(interactive)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    table='\n'.join(f"| {'键鼠' if a['lane']=='keyboard' else 'Action prompt'} | {a['accepted_segments']} | {a['world_seconds']:.2f}s | 两段间真实状态与 RGB 首帧接续核对通过 |" for a in audits)
    report=f'''# Agentic World Model：双输入与真实视频闭环

## 组会先展示什么

直接播放 `meeting_demo.mp4`，左右分别是送入 H3 的白模视频与对应生成结果。随后展示 `pipeline.svg`，最后打开 `真实闭环记录.md`，讲清楚第一段被接受之后，如何修正状态并进入第二段。

`index.html` 把视频、流程图和记录整理在一个离线页面中。`keyboard_rgb.mp4` 和 `action_rgb.mp4` 是不加说明条的生成视频。

## 这次实际跑出的结果

| 输入方式 | 已接受段数 | 世界内时长 | 接续核对 |
|---|---:|---:|---|
{table}

每段为 73 帧、24 FPS，首帧时间为 0、末帧为 3 秒。合并视频只去掉重复的段首时间点，没有用插帧、转场或运动修补掩盖边界。视频里的 ACCEPT 是实际 agent 审查与数据库提交结果。

这次使用的是 GTA 数据集中已有的场景几何与相机标定，再补充粗角色状态。不是仅凭一张图从零重建场景。主角和 NPC 使用与数据集风格接近的粗人形占位，步态和材质细节由 H3 补全。

独立浏览器实测另外生成了两次，均因几何对齐不充分被 GPT 拒绝。第二次的末尾步态已有改善，但没有强行接受；世界仍保持原有版本和历史。成功片与失败片都在交付包内。浏览器拒绝后恢复操作、已接受视频回放及下一输入窗口检查均通过。

## 两种交互怎样进入同一个世界

**键鼠**：WASD 与鼠标增量按 24 Hz 记录，由确定性控制器计算角色移动、相机跟随、加减速、落地和接触。GPT 不替换用户的按键轨迹，负责必要的世界扩展、生成提示词、视频审查和状态维护。展示视频的两段输入是可复现的脚本化键鼠记录；另外也通过浏览器真实按键和鼠标事件提交了独立的生成任务，记录见 `ui_review/`。

**Action prompt**：用户描述“沿砖路快步向前，和前方行人保持距离，稍向右侧让开，镜头平稳跟随”。GPT 将其拆成短时间窗的粗运动计划，并调用同一组 world 工具执行。段落边界不要求角色和相机停下。

两条入口都进入：已接受世界 → 计划 → 工具检查 → 编译白模与深度 → H3 → GPT 看实际视频 → 修正实际状态 → 提交历史。下一段使用提交后的状态与上一段真实末帧。

## GPT 与 H3 各负责什么

GPT 使用 Pi coding-agent harness 的文件工具和注册 world 工具。世界 SQLite、观察记录、agent 会话分开保存。`plan` 生成的内容只是意图；只有视频审查后的 `commit` 才推进世界时间和历史。

H3 本轮收到当前 RGB 首帧、白模参考视频、强度 0.5 的深度条件，以及 GPT 写的动作、运镜和外观提示词。分辨率 1024×576，40 个去噪步，EasyCache 关闭。采用现有 Ref2VA / ControlNet 推理环境，本轮没有训练新模型。

当前外观延续依赖真实 RGB 末帧与文本描述。师姐准备训练的“历史 RGB 与白模配对 memory”没有被标成已经部署的能力。

## 世界扩展怎样接入

键鼠分支实际调用了 `world extend`，在当前画面之外增加了一个巡逻 NPC。它有 `AGENT_CREATED` 来源和行为；既有物体与已接受历史不被覆盖。这个 NPC 在本次主视频中没有出镜，因此这部分证明的是世界扩展与离屏行为已经进入世界状态，尚未证明新 NPC 出镜后的外观稳定性。

运动检查写在工具中：计划编译前重新执行校验；视频修正后若终态穿入障碍或不落地，会在提交前拒绝，避免下一段才暴露不可执行状态。合理的落地、接触、贴墙滑动并不等于错误。

## 从真实运行中看到了什么

第一段里主角外观和前行趋势保持，NPC 却没有完全服从白模计划。GPT 在审查中记录了 NPC 的实际转向或停步，修正了相应状态。文字分支第一段被作为“部分完成”接受，再继续余下动作。这是闭环比单次生成多做的一步。

位置与速度目前仍是粗粒度的视频解释，不是毫米级测量。细小招牌文字、脚步接触和 NPC 运动仍可能漂移。多段结果和独立视觉复核结论见 `视觉复核.md`；原始视频、审查 JSON、提交回执和数据库快照均保留。

## 复现与文件

- `pipeline.svg` / `pipeline.png`：可编辑流程图与插入幻灯片的图片。
- `demo_keyboard.mp4` / `demo_action.mp4`：各分支白模与生成视频对照。
- `真实闭环记录.md` / `真实工具调用.json`：真实审查内容和工具调用顺序。
- `delivery_audit.json`：状态接续、首帧来源和回执核对。
- `evidence/`：两路实验视频、提示词、计划、修正、历史快照。
- `ui_review/`：浏览器桌面、手机端、真实提交截图与输入记录。

项目路径：`C:\\Users\\zjc\\omgd\\agentic_world`。服务器项目：`/home/star/agentic_world`。本轮服务器实验：`outputs/meeting_demo`。

服务器上的双输入入口是 `scripts/serve_controls.py --db <独立世界数据库> --gpu <空闲GPU> --port 8768`；本地通过 `python scripts/controls_tunnel.py` 连接。真实生成是分段异步过程，当前单个 3 秒视频约需 20 分钟级推理，不能按实时游戏延迟来介绍。
'''
    write(DEST/'组会汇报.md',report)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic World Model · 组会演示</title><style>
*{box-sizing:border-box}body{margin:0;font:16px/1.7 system-ui,sans-serif;color:#1b2925;background:#f6f8f7;letter-spacing:0}header,main{max-width:1320px;margin:auto;padding:22px 30px}header{border-bottom:1px solid #cbd5cf}h1{font-size:28px;margin:0}h2{font-size:21px;margin:28px 0 10px}p{margin:8px 0;color:#46554e}video{display:block;width:100%;max-height:72vh;background:#111}nav{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}button{font:inherit;padding:7px 16px;border:1px solid #bbc8c0;border-radius:4px;background:white;cursor:pointer}button[aria-pressed=true]{background:#146b50;color:white;border-color:#146b50}a{color:#146b50}img{width:100%;height:auto}section{border-top:1px solid #cbd5cf;margin-top:28px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;border-bottom:1px solid #d8dfdb;padding:10px}footer{padding:24px 0;color:#59675f;font-size:14px}@media(max-width:600px){header,main{padding:16px}h1{font-size:23px}button{padding:6px 10px}}</style>
<header><h1>Agentic World Model</h1><p>两种交互方式，同一个持续维护的世界。</p></header><main>
<nav aria-label="视频选择"><button data-src="meeting_demo.mp4" aria-pressed="true">完整演示</button><button data-src="demo_keyboard.mp4" aria-pressed="false">键鼠对照</button><button data-src="demo_action.mp4" aria-pressed="false">动作提示词对照</button><button data-src="keyboard_rgb.mp4" aria-pressed="false">键鼠原片</button><button data-src="action_rgb.mp4" aria-pressed="false">提示词原片</button></nav>
<video id="player" controls playsinline preload="metadata" src="meeting_demo.mp4" poster="input.png"></video>
<p>左侧：输入白模。右侧：真实 H3 输出。两段之间使用已接受实际状态与上一段 RGB 末帧接续。</p>
<section><h2>实际记录</h2><table><thead><tr><th>输入</th><th>已接受段数</th><th>世界时长</th></tr></thead><tbody>ROWS</tbody></table>
<p><a href="组会汇报.md">组会汇报</a> · <a href="真实闭环记录.md">GPT 审查与修正记录</a> · <a href="真实工具调用.json">真实工具调用</a> · <a href="视觉复核.md">视觉复核</a> · <a href="delivery_audit.json">接续核对</a></p></section>
<section><h2>一次真实修正</h2><img src="closed_loop_evidence.png" alt="计划末态、实际视频末帧、修正后白模与下一段首帧的对照"></section>
UI_SECTION
<section><h2>Pipeline</h2><a href="pipeline.svg"><img src="pipeline.svg" alt="双输入进入白模计划、H3 生成、GPT 视频审查、状态修正与历史提交的闭环"></a></section>
<footer>输入是 GTA 数据集几何与相机标定。键鼠主片使用可复现输入记录；生成按段异步进行。当前记忆训练尚未在本次 checkpoint 中部署。</footer></main>
<script>const video=document.getElementById('player');document.querySelectorAll('button[data-src]').forEach(button=>button.onclick=()=>{document.querySelectorAll('button[data-src]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));video.src=button.dataset.src;video.play().catch(()=>{});});</script></html>'''
    rows=''.join(f"<tr><td>{'键鼠' if a['lane']=='keyboard' else 'Action prompt'}</td><td>{a['accepted_segments']}</td><td>{a['world_seconds']:.2f}s</td></tr>" for a in audits)
    recordings=list((SOURCE/'ui_review/recording').glob('*.webm'))
    ui=''
    if recordings:
        path='ui_review/recording/'+recordings[-1].name
        ui=f'<section><h2>浏览器真实输入记录</h2><video controls playsinline preload="metadata" src="{html.escape(path)}"></video><p>自动化浏览器按键与鼠标事件，录制到真实任务提交。独立生成及重试均被 GPT 拒绝，未推进世界；结果与审查记录保留在 evidence/interactive。拒绝后可从原接受状态继续输入。</p></section>'
    write(DEST/'index.html',page.replace('ROWS',rows).replace('UI_SECTION',ui))
    if (SOURCE/'visual_review.md').exists():shutil.copy2(SOURCE/'visual_review.md',DEST/'视觉复核.md')
    print('Prepared',DEST)


if __name__=='__main__':main()
