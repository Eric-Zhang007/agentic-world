"""Concatenate only committed videos, in authoritative history order."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import av
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/gta_pi_loop_20261102'
REMOTE='/home/star/agentic_world/outputs/gta_pi_loop_20261102/'


def local(path):
    if not str(path).startswith(REMOTE):raise ValueError('Unexpected trial artifact path')
    return BASE/str(path)[len(REMOTE):]


def export(name):
    folder=BASE/name
    with closing(sqlite3.connect((folder/'world.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
        row=db.execute("SELECT value FROM world_meta WHERE key='whitebox_history'").fetchone()
    history=json.loads(row[0]) if row else []
    if not history:return
    first=local(history[0]['artifacts']['video'])
    with av.open(str(first)) as source:
        spec=source.streams.video[0];width,height,fps=spec.width,spec.height,spec.average_rate
    outputs=[folder/'accepted_rollout.mp4',folder/'accepted_comparison.mp4']
    with av.open(str(outputs[0]),'w') as rgb_out,av.open(str(outputs[1]),'w') as pair_out:
        rgb_stream=rgb_out.add_stream('libx264',rate=fps)
        pair_stream=pair_out.add_stream('libx264',rate=fps)
        for stream,w in [(rgb_stream,width),(pair_stream,width*2)]:
            stream.width=w;stream.height=height+32;stream.pix_fmt='yuv420p';stream.options={'crf':'19'}
        total=0
        for index,record in enumerate(history):
            video=local(record['artifacts']['video'])
            proxy=video.parent.parent/'compiled/whitebox.mp4'
            with av.open(str(video)) as rgb,av.open(str(proxy)) as white:
                spec=rgb.streams.video[0]
                if (spec.width,spec.height,spec.average_rate)!=(width,height,fps):
                    raise ValueError('Accepted segments have incompatible video formats')
                for frame_index,(observed,planned) in enumerate(zip(rgb.decode(video=0),white.decode(video=0),strict=True)):
                    if index and frame_index==0:continue
                    label=f"Segment {index+1} | {'action complete' if record['action_complete'] else 'partial action'} | world {record['start']+frame_index/float(fps):.3f}s"
                    rgb_image=Image.new('RGB',(width,height+32),'#202020')
                    rgb_image.paste(observed.to_image(),(0,32));ImageDraw.Draw(rgb_image).text((8,9),label,fill='white')
                    paired=Image.new('RGB',(width*2,height+32),'#202020')
                    paired.paste(planned.to_image().resize((width,height)),(0,32));paired.paste(rgb_image,(width,0))
                    ImageDraw.Draw(paired).text((8,9),'Planned whitebox | '+label,fill='white')
                    for stream,container,image in [(rgb_stream,rgb_out,rgb_image),(pair_stream,pair_out,paired)]:
                        for packet in stream.encode(av.VideoFrame.from_image(image)):container.mux(packet)
                    total+=1
        for stream,container in [(rgb_stream,rgb_out),(pair_stream,pair_out)]:
            for packet in stream.encode():container.mux(packet)
    result={'case':name,'accepted_segments':len(history),'frames':total,
            'elapsed_world_sec':history[-1]['end']-history[0]['start'],
            'boundary_rule':'Omit repeated boundary timestamp; generated first frames are not pixel-identical to prior endpoints',
            'paths':[str(p) for p in outputs]}
    (folder/'accepted_rollout_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['urban','harbor','rural','rural/continuation','rural/road_edge','urban/npc_walk'])
    args=parser.parse_args()
    cases=[args.case] if args.case else ['urban','harbor','rural']
    if not args.case and (BASE/'rural/continuation/world.sqlite').exists():cases.append('rural/continuation')
    for case in cases:export(case)
