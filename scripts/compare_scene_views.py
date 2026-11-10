"""Put actual reference and coarse previews on the same output aspect ratio."""
import argparse
from pathlib import Path

from PIL import Image,ImageDraw


def compare(images, labels, output):
    if len(images)!=len(labels) or not images:
        raise ValueError('Each image needs a label')
    canvas=Image.new('RGB',(512*len(images),412),'#181818')
    draw=ImageDraw.Draw(canvas)
    for index,(image,label) in enumerate(zip(images,labels)):
        # The existing H3 compiler stretches the 960x640 Godot image to 1024x768.
        canvas.paste(Image.open(image).convert('RGB').resize((512,384)),(512*index,28))
        draw.text((512*index+8,8),label,fill='white')
    canvas.save(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,action='append',required=True)
    p.add_argument('--label',action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.image,a.label,a.output)
