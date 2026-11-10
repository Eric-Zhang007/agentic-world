"""Create timestamped visual evidence sheets from local historical videos."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/feishu_20261013'
VIDEOS={
    'early_11s':'action_wukong_outline_20261008/first_action_11s.mp4',
    'posture_00':'posture_live_20261010/segment_00/render/controlled.mp4',
    'posture_01':'posture_live_20261010/segment_01/render/controlled.mp4',
    'memory_hold':'appearance_memory_20261012/generated.mp4',
    'refined':'refined_rollouts_20261012/segment_00.mp4',
    'continuous':'continuous_motion_20261012/segment_00.mp4',
    'natural_walk':'natural_walk_dolly_20261012/generated.mp4',
    'orbit_baseline':'identity_repair_20261013/prompt_regression_run1/baseline/generated.mp4',
}


def main():
    (OUT/'images').mkdir(parents=True,exist_ok=True)
    for name,relative in VIDEOS.items():
        cap=cv2.VideoCapture(str(ROOT/'outputs'/relative))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not count or not fps:
            raise ValueError(relative)
        sheet=Image.new('RGB',(1200,648),'white');draw=ImageDraw.Draw(sheet)
        for i,fraction in enumerate([0,.08,.25,.5,.75,1]):
            frame=round((count-1)*fraction)
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,image=cap.read()
            if not ok:raise ValueError((relative,frame))
            image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            image.thumbnail((400,300))
            x=i%3*400;y=i//3*324
            sheet.paste(image,(x+(400-image.width)//2,y+24))
            draw.text((x+8,y+5),f'{name} | {frame/fps:.2f}s',fill='black')
        cap.release()
        sheet.save(OUT/'images'/f'{name}.jpg',quality=93)
        print(name,count,fps)


if __name__=='__main__':main()
