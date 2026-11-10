"""Export synchronized planned/observed videos and timestamped contact sheets."""
import argparse
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/gta_pi_loop_20261102'


def export(video):
    attempt=video.parent.parent
    planned=attempt/'compiled/whitebox.mp4'
    if not planned.exists():return
    output=attempt/'comparison.mp4'
    samples=[];paired=[]
    with av.open(str(video)) as observed,av.open(str(planned)) as proxy,av.open(str(output),'w') as destination:
        source=observed.streams.video[0];fps=source.average_rate
        width,height=source.width,source.height
        stream=destination.add_stream('libx264',rate=fps)
        stream.width=width*2;stream.height=height+32;stream.pix_fmt='yuv420p'
        stream.options={'crf':'20'}
        count=0
        for index,(rgb,white) in enumerate(zip(observed.decode(video=0),proxy.decode(video=0),strict=True)):
            panel=Image.new('RGB',(width*2,height+32),'#202020')
            panel.paste(white.to_image().resize((width,height)),(0,32));panel.paste(rgb.to_image(),(width,32))
            draw=ImageDraw.Draw(panel)
            draw.text((8,9),f'Planned whitebox | {index/float(fps):.3f}s',fill='white')
            draw.text((width+8,9),'H3 observed RGB',fill='white')
            for packet in stream.encode(av.VideoFrame.from_image(panel)):destination.mux(packet)
            if index%12==0 or index==source.frames-1:
                samples.append((index/float(fps),rgb.to_image()))
                paired.append((index/float(fps),white.to_image(),rgb.to_image()))
            count+=1
        for packet in stream.encode():destination.mux(packet)
    columns=4;rows=(len(samples)+columns-1)//columns
    sheet=Image.new('RGB',(columns*320,rows*204),'#202020');draw=ImageDraw.Draw(sheet)
    for i,(timestamp,image) in enumerate(samples):
        image.thumbnail((320,180));x=i%columns*320;y=i//columns*204
        sheet.paste(image,(x,y));draw.text((x+5,y+183),f'{timestamp:.3f}s',fill='white')
    sheet.save(attempt/'observed_frames.jpg',quality=94)
    pair_sheet=Image.new('RGB',(640,len(paired)*204),'#202020')
    draw=ImageDraw.Draw(pair_sheet)
    for index,(timestamp,white,rgb) in enumerate(paired):
        for column,image in enumerate((white,rgb)):
            image.thumbnail((320,180));pair_sheet.paste(image,(column*320,index*204))
        draw.text((5,index*204+183),f'{timestamp:.3f}s | planned whitebox / observed RGB',fill='white')
    pair_sheet.save(attempt/'paired_frames.jpg',quality=94)
    print(str(output),count)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['urban','harbor','rural','rural/continuation','rural/road_edge','urban/npc_walk']);a=p.parse_args()
    root=BASE/a.case if a.case else BASE
    for video in sorted(root.rglob('controlled.mp4')):
        if video.parent.name=='render':export(video)


if __name__=='__main__':main()
