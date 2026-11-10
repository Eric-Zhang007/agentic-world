"""Compile an agent-authored action and save RGB/depth review sheets."""
import argparse
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness/pi'))
from world_tools import dispatch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=dispatch('compile',{'plan':str(a.plan),'output':str(a.output),'width':768,'height':432})
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    indices=[None,11,23,35,47,59,71]
    sheet=Image.new('RGB',(384*len(indices),2*240),'white');draw=ImageDraw.Draw(sheet)
    for i,index in enumerate(indices):
        name='camera.png' if index is None else f'frame_{index:06d}.png'
        for j,kind in enumerate(('proxy','depth')):
            image=Image.open(a.output/kind/name).convert('RGB').resize((384,216))
            sheet.paste(image,(i*384,j*240+24))
            draw.text((i*384+6,j*240+5),f'{kind} | {0 if index is None else (index+1)/24:.2f}s',fill='black')
    sheet.save(a.output/'review.jpg',quality=95)
    print(json.dumps(result))


if __name__=='__main__':main()
