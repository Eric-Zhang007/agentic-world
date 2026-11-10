"""Package existing, verified experiment artifacts for the group meeting."""
import json
import shutil
import subprocess
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt
from docx.oxml.ns import qn
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('C:/Users/zjc/Desktop/AgenticWorld组会材料')


def main():
    mapping={
        'videos/01_whitebox_depth_comparison.mp4':'outputs/native_whitebox_reference_20261013/native_depth_comparison.mp4',
        'videos/02_partial_action_continuation.mp4':'outputs/posture_live_20261010/accepted_sequence.mp4',
        'videos/03_first_segment_whitebox_comparison.mp4':'outputs/posture_live_20261010/segment_00_accepted_comparison.mp4',
        'videos/04_second_segment_whitebox_comparison.mp4':'outputs/posture_live_20261010/segment_01_accepted_comparison.mp4',
        'images/posture_transition.jpg':'outputs/posture_live_20261010/segment_00_posture_reconcile.jpg',
        'images/continuation_frames.jpg':'outputs/posture_live_20261010/segment_01_frames.jpg',
        'images/boundary_continuity.jpg':'outputs/posture_live_20261010/boundary_continuity.jpg',
        'evidence/posture_chain.json':'outputs/posture_live_20261010/chain_verification.json',
        'evidence/posture_state.json':'outputs/posture_live_20261010/posture_state_report.json',
        'evidence/native_video_readme.md':'outputs/native_whitebox_reference_20261013/README.md',
    }
    for dest,source in mapping.items():
        target=OUT/dest;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/source,target)
    shutil.copy2(Path('C:/Users/zjc/Desktop/GTA相机轨迹与帧对/examples/01_trajectory_and_pair.jpg'),OUT/'images/camera_pair.jpg')
    # A first/middle/final sheet allows direct review of both actual videos.
    checks=[]
    for name in [n for n in mapping if n.startswith('videos/')]:
        path=OUT/name
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
        duration=float(info['format']['duration'])
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True)
        shots=[]
        for i,t in enumerate((0,duration/2,max(0,duration-.1))):
            frame=OUT/'images'/f'{path.stem}_{i}.jpg'
            subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',str(path),'-frames:v','1','-update','1',str(frame)],check=True,capture_output=True)
            im=Image.open(frame).convert('RGB');im.thumbnail((640,480));shots.append(im)
        sheet=Image.new('RGB',(640*3,480),'white')
        for i,im in enumerate(shots):sheet.paste(im,(640*i,0))
        sheet.save(OUT/'images'/f'{path.stem}_overview.jpg',quality=92)
        checks.append({'path':name,'duration_sec':duration,'full_decode':True,'source':mapping[name]})
    (OUT/'evidence/media_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    doc=Document()
    doc.sections[0].left_margin=doc.sections[0].right_margin=Cm(2)
    style=doc.styles['Normal'];style.font.size=Pt(10.5);style.font.name='Calibri'
    style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    code=False
    for line in (OUT/'组会可直接粘贴.md').read_text(encoding='utf-8').splitlines():
        if line.startswith('```'):code=not code;continue
        if not line.strip():continue
        if line.startswith('# '):doc.add_heading(line[2:],0)
        elif line.startswith('## '):doc.add_heading(line[3:],1)
        else:
            p=doc.add_paragraph(line.replace('**',''))
            if code:
                for r in p.runs:r.font.name='Consolas';r.font.size=Pt(9)
        if line.startswith('**配图：** images/camera_pair'):
            doc.add_picture(str(OUT/'images/camera_pair.jpg'),width=Cm(16.5))
        elif line.startswith('**配图：** images/posture'):
            doc.add_picture(str(OUT/'images/posture_transition.jpg'),width=Cm(16.5))
    doc.save(OUT/'组会汇报.docx')
    sections=[]
    for name in [n for n in mapping if n.startswith('videos/')]:
        sections.append(f'<h2>{name}</h2><video controls preload="metadata" src="{name}" style="width:min(100%,1100px)"></video>')
    (OUT/'演示视频.html').write_text('<!doctype html><meta charset="utf-8"><title>组会演示</title><body style="font:16px system-ui;max-width:1200px;margin:24px auto;padding:12px"><h1>组会演示素材</h1>'+''.join(sections)+'</body>',encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
