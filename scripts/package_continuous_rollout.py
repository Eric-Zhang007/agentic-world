"""Create reviewable sequence and seam artifacts without changing world state."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from renderer.continuation import read_video,save_video


def labeled(pixels,label,width=512):
    source=Image.fromarray(pixels);height=round(source.height*width/source.width)
    image=Image.new('RGB',(width,height+32),'black');image.paste(source.resize((width,height)),(0,32))
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    except OSError:font=ImageFont.load_default()
    ImageDraw.Draw(image).text((8,8),label,fill='white',font=font)
    return image


def package(root):
    accepted=[];attempts=[]
    for segment in sorted(root.glob('segment_*')):
        status=segment/'review/status.json'
        if not status.exists():continue
        decision=json.loads(status.read_text())['status']
        video=segment/'continuation.mp4'
        if not video.exists():video=segment/'render/controlled.mp4'
        pixels=read_video(video);attempts.append((segment,pixels,decision))
        if decision=='COMMITTED':accepted.append((segment,pixels))
        indices=[0,len(pixels)//2,len(pixels)-1]
        tiles=[labeled(pixels[i],f'{segment.name} {decision} t={i/24:.3f}s') for i in indices]
        sheet=Image.new('RGB',(sum(t.width for t in tiles),tiles[0].height))
        for i,tile in enumerate(tiles):sheet.paste(tile,(i*tile.width,0))
        sheet.save(root/(segment.name+'_frames.jpg'),quality=94)
    if not attempts:return {'accepted_segments':0}
    visual=[]
    for segment,pixels,decision in attempts:
        visual.extend(np.asarray(labeled(frame,f'{segment.name} {decision} (experiment)',768)) for frame in pixels)
    save_video(np.stack(visual),root/'experiment_sequence_labeled.mp4')
    if accepted:
        save_video(np.concatenate([pixels if i==0 else pixels[1:] for i,(_,pixels) in enumerate(accepted)]),root/'accepted_sequence.mp4')
    if len(attempts)>=2:
        (_,first,status0),(_,second,status1)=attempts[:2]
        seam=np.concatenate([first[-24:],second[1:25]])
        annotated=np.stack([np.asarray(labeled(frame,'EXPERIMENT SEAM: '+status0+' -> '+status1,768)) for frame in seam])
        save_video(np.repeat(annotated,2,axis=0),root/'boundary_half_speed.mp4')
        tiles=[labeled(first[i],f'Prior {status0} {i/24:.3f}s',384) for i in [len(first)-13,len(first)-7,len(first)-1]]
        tiles += [labeled(second[i],f'Next {status1} {i/24:.3f}s',384) for i in [0,6,12]]
        sheet=Image.new('RGB',(3*tiles[0].width,2*tiles[0].height))
        for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*tile.width,(i//3)*tile.height))
        sheet.save(root/'boundary_continuity.jpg',quality=94)
    result={'accepted_segments':len(accepted),'duplicate_boundary_frames_removed':max(0,len(accepted)-1),
            'seam_half_speed_is_visualization_only':True,'world_edited':False}
    (root/'visualization_metadata.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(package(p.parse_args().root),indent=2))
