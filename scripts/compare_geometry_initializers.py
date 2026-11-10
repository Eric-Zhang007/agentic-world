"""Same-source visual contact sheet, preserving each panel's aspect ratio."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', action='append', required=True, metavar='LABEL=PATH')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    width, height, header = 960, 544, 34
    rows = (len(a.image)+1)//2
    canvas = Image.new('RGB', (width*2, (height+header)*rows), '#202020')
    draw = ImageDraw.Draw(canvas)
    for i, item in enumerate(a.image):
        label, path = item.split('=', 1)
        with Image.open(path) as source:
            panel = ImageOps.contain(source.convert('RGB'), (width, height), Image.Resampling.LANCZOS)
        x, y = (i%2)*width, (i//2)*(height+header)
        draw.text((x+12, y+10), label, fill='white')
        canvas.paste(panel, (x+(width-panel.width)//2, y+header+(height-panel.height)//2))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.output, quality=95)
    print(a.output)


if __name__ == '__main__':
    main()
