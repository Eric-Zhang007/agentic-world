"""Lay out saved plan, real RGB, corrected world and continuation evidence."""
import json
from pathlib import Path
import math
from PIL import Image,ImageDraw,ImageFont
from scripts.package_meeting_demo import frames

root=Path(__file__).resolve().parents[1]/'outputs/meeting_demo'
first=root/'action/segment_01/attempt_01'
second=root/'action/segment_02/attempt_01'
review=json.loads((first/'review.json').read_text(encoding='utf-8'))
def font(size):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
canvas=Image.new('RGB',(1600,1220),'white');draw=ImageDraw.Draw(canvas)
draw.text((32,24),'接受之后，先修正实际状态，再接下一段',font=font(34),fill='#1c2925')
draw.text((32,78),'真实记录示例：NPC 朝向偏离计划；接受部分进展，保留剩余动作目标。',font=font(22),fill='#52635b')
pictures=[frames(first/'compiled/whitebox.mp4')[-1],Image.open(first/'evidence/frames/000072.png'),
          Image.open(first/'correction_preview/camera.png'),Image.open(second/'evidence/frames/000000.png')]
titles=['1  原计划末态白模','2  实际 H3 视频末帧','3  按视频审查提交后的白模','4  下一段真实视频首帧']
notes=['NPC 原计划为侧向运动。','人物身份保持，NPC 实际转向街道深处。',
       '保存朝向修正，历史保留原计划与接受解释。','使用上一段 RGB 末帧；从已提交世界状态规划。']
for i,(picture,title,note) in enumerate(zip(pictures,titles,notes)):
    x=32+(i%2)*784;y=138+(i//2)*500
    draw.text((x,y),title,font=font(24),fill='#126c51')
    canvas.paste(picture.convert('RGB').resize((752,423)),(x,y+42))
    draw.text((x,y+466),note,font=font(16),fill='#53625b')
offset=review['corrections'][0]['keyframes'][-1]['yaw_offset']
draw.text((32,1170),f'本例实际写入的 NPC 朝向偏移：{offset:.2f} rad（{math.degrees(offset):.1f}°）；属于粗粒度视频估计。',font=font(19),fill='#3b4b42')
canvas.save(root/'closed_loop_evidence.png')
print(root/'closed_loop_evidence.png')
