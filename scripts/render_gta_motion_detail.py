"""Inspect a fixed image region at dense timestamps without motion interpolation."""
import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--box',type=int,nargs=4,required=True,metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    parser.add_argument('--interval',type=float,default=.125)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.interval<=0:parser.error('interval must be positive')
    samples=[];next_time=0.
    with av.open(str(args.video)) as container:
        stream=container.streams.video[0]
        left,top,right,bottom=args.box
        if not (0<=left<right<=stream.width and 0<=top<bottom<=stream.height):
            parser.error('box must be inside the video')
        origin=None
        for frame in container.decode(video=0):
            if frame.pts is None:raise ValueError('Timestamp required')
            timestamp=float(frame.pts*frame.time_base)
            if origin is None:origin=timestamp
            elapsed=timestamp-origin
            if elapsed+1e-7>=next_time:
                samples.append((elapsed,frame.to_image().crop(args.box)))
                next_time+=args.interval
    if not samples:raise ValueError('Empty video')
    columns=8;rows=(len(samples)+columns-1)//columns
    sheet=Image.new('RGB',(columns*160,rows*220),'#202020')
    draw=ImageDraw.Draw(sheet)
    for index,(timestamp,crop) in enumerate(samples):
        scale=min(160/crop.width,192/crop.height)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.NEAREST)
        x=(index%columns)*160;y=(index//columns)*220
        sheet.paste(crop,(x+(160-crop.width)//2,y))
        draw.text((x+5,y+197),f'{timestamp:.3f}s',fill='white')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(args.output,quality=95)
    print(str(args.output))


if __name__=='__main__':main()
