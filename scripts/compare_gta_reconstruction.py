"""Render held-out dataset comparison after image-only initialization finishes."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gta_pi_loop_20261102'

def main():
    sheet=Image.new('RGB',(1920,1170),'#202020')
    draw=ImageDraw.Draw(sheet)
    for row,name in enumerate(['rural','urban','harbor']):
        base=OUT/name
        report=json.loads((base/'reconstruction_report.json').read_text())
        selected=report.get('chosen_preview','preview_03')
        if isinstance(selected,dict):selected=selected.get('path',selected.get('preview'))
        selected=Path(selected)
        if selected.suffix=='.png':preview=base.joinpath(*selected.parts[selected.parts.index(name)+1:]) if name in selected.parts else base/selected
        else:preview=base/selected.name/'views/camera.png'
        if not preview.exists():
            previews=sorted(base.glob('preview_*/views/camera.png'))
            preview=previews[-1]
        for col,(label,path) in enumerate([('RGB input',base/'input.png'),
                ('Dataset whitebox (held out)',base/'dataset_whitebox.png'),('Pi RGB-only whitebox',preview)]):
            image=Image.open(path).convert('RGB');image.thumbnail((640,360))
            x,y=col*640,row*390
            sheet.paste(image,(x,y))
            draw.text((x+8,y+364),name+' | '+label,fill='white')
    sheet.save(OUT/'reconstruction_comparison.jpg',quality=96)
    print(OUT/'reconstruction_comparison.jpg')

if __name__=='__main__':main()
