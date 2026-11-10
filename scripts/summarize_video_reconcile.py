"""Build inspectable planned/accepted/video artifacts for the real loop."""
import argparse
import json
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_comfy_control import save_video


def main(root):
    from state.db import WorldDB
    from scripts.run_video_reconcile_trial import verify_chain
    db=WorldDB(root/'world.sqlite')
    try:verify_chain(db,root)
    finally:db.connection.close()
    result=[];accepted_clips=[];rollout_clips=[]
    for segment in sorted(root.glob('segment_*')):
        if not (segment/'review/status.json').exists():continue
        status=json.loads((segment/'review/status.json').read_text())
        review=json.loads((segment/'review/review.json').read_text())
        metadata=json.loads((segment/'render/metadata.json').read_text())
        row={'segment':segment.name,'status':status,'review':review,
             'native_guide_indices':metadata.get('resolved_guide_indices'),
             'sample_seconds':metadata['runs'][0]['sample_seconds']}
        result.append(row)
        with av.open(str(segment/'render'/(metadata['runs'][0]['label']+'.mp4'))) as container:
            generated=[frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        label=segment.name+(' ACCEPTED / '+('ACTION COMPLETE' if status['action_complete'] else 'PARTIAL ACTION')
                           if status['status']=='COMMITTED' else ' REJECTED / NOT COMMITTED')
        rollout_clips.append((generated,label))
        sheet=Image.new('RGB',(4*384,3*312),'white');draw=ImageDraw.Draw(sheet)
        indices=np.rint(np.linspace(0,len(generated)-1,12)).astype(int)
        for cell,index in enumerate(indices):
            x,y=cell%4*384,cell//4*312
            sheet.paste(Image.fromarray(generated[index]).resize((384,288)),(x,y+24))
            draw.text((x+4,y+4),f'frame {index} | {index/24:.2f}s | {status["status"]}',fill='black')
        sheet.save(root/(segment.name+'_frames.jpg'))
        if status['status']!='COMMITTED':continue
        accepted_clips.append(generated)
        corrected_files=[segment/'corrected_preview/camera.png']+sorted((segment/'corrected_preview').glob('frame_*.png'))
        assert len(corrected_files)==len(generated)
        panels=[]
        for index,pixels in enumerate(generated):
            corrected=Image.open(corrected_files[index]).convert('RGB').resize((512,384))
            video=Image.fromarray(pixels).resize((512,384))
            panel=Image.new('RGB',(1024,412),'black');draw=ImageDraw.Draw(panel)
            panel.paste(video,(0,28));panel.paste(corrected,(512,28))
            draw.text((8,8),f'{segment.name} actual H3 | t={index/24:.2f}',fill='white')
            draw.text((520,8),'Accepted corrected coarse world (agent estimate)',fill='white')
            panels.append(np.asarray(panel))
        save_video(np.stack(panels).astype(np.float32)/255,root/(segment.name+'_accepted_comparison.mp4'))
    if len(accepted_clips)>=2:
        save_video(np.concatenate(accepted_clips).astype(np.float32)/255,root/'accepted_sequence.mp4')
    if len(rollout_clips)>=2:
        labeled=[]
        for clip,label in rollout_clips:
            for pixels in clip:
                panel=Image.new('RGB',(pixels.shape[1],pixels.shape[0]+28),'black')
                panel.paste(Image.fromarray(pixels),(0,28))
                ImageDraw.Draw(panel).text((8,8),label,fill='white')
                labeled.append(np.asarray(panel))
        save_video(np.stack(labeled).astype(np.float32)/255,root/'rollout_sequence_labeled.mp4')
        seam=Image.new('RGB',(1536,412),'black');draw=ImageDraw.Draw(seam)
        for i,(pixels,label) in enumerate([
            (rollout_clips[0][0][-1],rollout_clips[0][1]+' final'),
            (rollout_clips[1][0][0],rollout_clips[1][1]+' first'),
            (rollout_clips[1][0][-1],rollout_clips[1][1]+' final')
        ]):
            seam.paste(Image.fromarray(pixels).resize((512,384)),(i*512,28))
            draw.text((i*512+8,8),label,fill='white')
        seam.save(root/'boundary_continuity.jpg')
    (root/'run_summary.json').write_text(json.dumps({'segments':result,'tests_are_not_video_evidence':True},indent=2))
    print(json.dumps({'segments':[{k:v for k,v in row.items() if k!='review'} for row in result]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root.resolve())
