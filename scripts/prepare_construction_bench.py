"""Stage RGB-only benchmark inputs with separate paired evaluation references."""
import argparse
import json
from pathlib import Path
import shutil
from PIL import Image, ImageOps, ImageDraw

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidates',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=json.loads(a.candidates.read_text())
a.output.mkdir(parents=True,exist_ok=False)
manifest=[]
tiles=[]
for row in rows:
    if not all(Path(row[k]).is_file() for k in ('rgb','whitebox','state')):
        continue
    name=f"case_{row['index']:02d}"
    case=a.output/'inputs'/name
    truth=a.output/'evaluation_only'/name
    case.mkdir(parents=True); truth.mkdir(parents=True)
    shutil.copy2(row['rgb'],case/'input.png')
    shutil.copy2(row['whitebox'],truth/'whitebox.png')
    shutil.copy2(row['state'],truth/'state.json')
    entry=dict(row,case=name,status='candidate_not_yet_reviewed')
    manifest.append(entry)
    tile=Image.new('RGB',(480,300),'white')
    with Image.open(row['rgb']) as im:
        tile.paste(ImageOps.contain(im.convert('RGB'),(480,270)),(0,30))
    ImageDraw.Draw(tile).text((10,8),name+' '+row['episode'],fill='black')
    tiles.append(tile)
sheet=Image.new('RGB',(480*3,300*max(1,(len(tiles)+2)//3)),'white')
for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*480,(i//3)*300))
sheet.save(a.output/'candidates.jpg')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'staged':len(manifest),'output':str(a.output)}))
