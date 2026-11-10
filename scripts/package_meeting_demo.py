"""Make presentation videos from accepted histories, without hiding boundaries."""
from contextlib import closing
import html
import json
from pathlib import Path
import sqlite3
import sys
import textwrap
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/meeting_demo'
REMOTE='/home/star/agentic_world/outputs/meeting_demo/'
FONT=Path('C:/Windows/Fonts/msyh.ttc')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def local(p):
    if not str(p).startswith(REMOTE):raise ValueError('Artifact is outside this demo: '+str(p))
    return OUT/str(p)[len(REMOTE):]
def font(size):return ImageFont.truetype(str(FONT),size)


def frames(path):
    with av.open(str(path)) as video:
        return [f.to_image().convert('RGB') for f in video.decode(video=0)]


def save_video(path,pictures,fps=24):
    with av.open(str(path),'w') as output:
        stream=output.add_stream('libx264',rate=fps);stream.width,stream.height=pictures[0].size
        stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'medium'}
        for picture in pictures:
            for packet in stream.encode(av.VideoFrame.from_image(picture)):output.mux(packet)
        for packet in stream.encode():output.mux(packet)


def text(draw,xy,value,size=22,fill='#edf3ef',width=1000):
    x,y=xy;line=''
    for ch in value:
        if draw.textlength(line+ch,font=font(size))>width:
            draw.text((x,y),line,font=font(size),fill=fill);y+=size+9;line=''
        line+=ch
    draw.text((x,y),line,font=font(size),fill=fill)
    return y+size+9


def lane(name):
    base=OUT/name
    with closing(sqlite3.connect((base/'delivery_world.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
        metadata={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM world_meta')}
    history=metadata.get('whitebox_history',[])
    if not history:raise ValueError(name+' has no accepted video')
    sheets=[];clean=[];checks=[];previous=None;last_rgb=None;previous_image=None
    title='键鼠输入 · GPT 维护世界' if name=='keyboard' else 'Action prompt · GPT 规划动作'
    for index,record in enumerate(history):
        artifact=record['artifacts'];video=local(artifact['video']);plan=local(artifact['planned_timeline']).parent
        command=read(video.parent/'generation_metadata.json')['command']
        reference=command[command.index('--reference-image')+1]
        proxy_path=local(command[command.index('--reference-video')+1])
        planned_scene=read(plan/'scene.json');after=read(local(artifact['corrected_timeline']).parent/'world_after.json')
        planned_end={e['id']:e for e in read(plan/'timeline.json')['frames'][-1]['entities']}
        motion_changed=[e['id'] for e in after['entities'] if e.get('motion',{}).get('velocity')
                        !=planned_end[e['id']].get('motion',{}).get('velocity')]
        checks.append({'segment':index+1,'revision_before':record['revision_before'],'revision_after':record['revision_after'],
            'video':str(video.relative_to(OUT)),'proxy':str(proxy_path.relative_to(OUT)),
            'starts_from_previous_actual_state':None if previous is None else planned_scene==previous,
            'uses_last_accepted_rgb':None if last_rgb is None else reference==last_rgb,
            'saved_receipt_matches_history':read(local(artifact['corrected_timeline']).parent/'receipt.json')==record,
            'actual_state_corrected':bool(motion_changed or record['visual_review'].get('corrections') or record['visual_review'].get('pose_corrections') or record['visual_review'].get('camera_correction')),
            'endpoint_motion_changed_ids':motion_changed,
            'action_complete':record['action_complete'],'geometry_diagnostics':record.get('geometry_diagnostics'),
            'review':record['visual_review']})
        if previous is not None and (planned_scene!=previous or reference!=last_rgb):
            raise ValueError('Accepted-state continuity failed for '+name)
        rgb=frames(video);proxy=frames(proxy_path)
        if len(rgb)!=len(proxy):raise ValueError('RGB and whitebox lengths differ')
        checks[-1]['decoded_rgb_frames']=len(rgb)
        checks[-1]['decoded_whitebox_frames']=len(proxy)
        if previous_image is not None:
            difference=np.abs(np.asarray(rgb[0],dtype=float)-np.asarray(previous_image,dtype=float))
            checks[-1]['boundary_rgb_mean_absolute_pixel_error']=float(difference.mean())
            checks[-1]['boundary_pixel_error_scope']='Decoded boundary equality, not a motion or identity metric'
            timeline=read(plan/'timeline.json')
            initial={e['id']:e for e in planned_scene['entities']}
            motion=[]
            for entity in timeline['frames'][0]['entities']:
                before=initial[entity['id']]
                if 'motion' not in before:continue
                velocity=(np.asarray(entity['position'])-np.asarray(before['position']))*timeline['fps']
                motion.append({'id':entity['id'],'accepted_velocity':before['motion']['velocity'],
                               'next_planned_first_tick_velocity':velocity.tolist()})
            checks[-1]['boundary_motion_state']=motion
        action=record['action'];review=record['visual_review']
        for tick,(image,white) in enumerate(zip(rgb,proxy)):
            if index and tick==0:continue
            canvas=Image.new('RGB',(1600,650),'#17201c');draw=ImageDraw.Draw(canvas)
            text(draw,(24,15),title,26,width=1200)
            text(draw,(1240,20),f"第 {index+1} 段 · {record['start']+tick/24:.2f}s",20,width=340)
            canvas.paste(white.resize((800,450)),(0,82));canvas.paste(image.resize((800,450)),(800,82))
            text(draw,(24,55),'执行的白模计划',18);text(draw,(824,55),'H3 原始生成画面',18)
            if action.get('type')=='controls':
                sample=action['inputs'][max(0,tick-1)];pressed=' + '.join(sample['keys']).upper() or '松开'
                line=f"输入：{pressed}    鼠标增量：{sample['mouse']}    24 Hz 确定性执行"
            else:line='动作意图：'+action.get('goal','')
            text(draw,(24,548),line,20,width=1540)
            verdict='动作完成' if record['action_complete'] else '接受实际进展，仍有后续动作'
            text(draw,(24,604),f"视频审查 → ACCEPT → 修正并提交世界版本 {record['revision_after']} ｜ {verdict}",18,fill='#a9dfbf',width=1540)
            sheets.append(canvas);clean.append(image)
        previous=after;last_rgb=artifact['last_frame'];previous_image=rgb[-1]
    save_video(OUT/f'demo_{name}.mp4',sheets)
    save_video(OUT/f'{name}_rgb.mp4',clean)
    audit={'lane':name,'accepted_segments':len(history),'world_seconds':metadata['tick'],
           'checks':checks,'extensions':metadata.get('whitebox_reconciliations',[]),
           'boundary_edit':'Only duplicate initial timestamps removed; no interpolation, crossfade or motion repair'}
    (OUT/f'{name}_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return sheets,audit


def main():
    all_frames=[];audits=[]
    for name in ('keyboard','action'):
        pictures,audit=lane(name);audits.append(audit)
        card=Image.new('RGB',(1600,650),'#17201c');d=ImageDraw.Draw(card)
        title='键鼠：确定性运动 + GPT 世界维护' if name=='keyboard' else '文字：GPT 规划 + 视频反馈修正'
        text(d,(75,180),title,40,width=1440)
        text(d,(75,260),'真实 H3 输出 · 对应白模条件 · 接受后接续',27,width=1440)
        detail='脚本化 WASD / 鼠标输入；不是实测实时游戏延迟。' if name=='keyboard' else '动作：沿砖路快步前进，避让行人，镜头连续跟随。'
        text(d,(75,330),detail,24,width=1440)
        text(d,(75,410),'初始世界使用 GTA 数据集几何与相机标定。',22,width=1440)
        all_frames += [card]*60+pictures+[pictures[-1]]*24
    save_video(OUT/'meeting_demo.mp4',all_frames)
    (OUT/'delivery_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{'lane':a['lane'],'accepted_segments':a['accepted_segments'],'world_seconds':a['world_seconds']} for a in audits]))


if __name__=='__main__':main()
