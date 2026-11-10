"""Reproducible visual spot check of previously unreviewed, unflagged records."""
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / 'outputs/dataset_reaudit'
OUT = ROOT / 'normal_random_review'
ARCHIVE = Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip')

def main():
    OUT.mkdir(exist_ok=True)
    ledger = [json.loads(s) for s in (ROOT/'review_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    eligible = [r for r in ledger if not r['reasons'] and r['visual_review']['status'] == 'NOT_DIRECTLY_REVIEWED' and r['reference_visual_review']['status'] == 'NOT_DIRECTLY_REVIEWED']
    chosen = random.Random(20261103).sample(sorted(eligible, key=lambda r:r['sample_id']), 24)
    meta = {r['sample_id']:r for r in map(json.loads,(ROOT/'metadata/records.jsonl').read_text(encoding='utf-8').splitlines())}
    manifest = {'seed':20261103,'method':'Simple random sample without replacement of unflagged previously unreviewed records','eligible':len(eligible),'count':len(chosen),'tasks':dict(Counter(r['task'] for r in chosen)),'episodes':dict(Counter(r['source'][0] for r in chosen)),'splits':dict(Counter(r['split'] for r in chosen)),'records':chosen}
    (OUT/'selection.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE) as z:
        for page in range(6):
            target = Image.new('RGB',(1280,1880),'white')
            refs = Image.new('RGB',(1280,840),'white')
            for slot,r in enumerate(chosen[page*4:page*4+4]):
                n=page*4+slot
                sid=r['sample_id']
                contact=Image.open(ROOT/'media/contact'/f'{sid}.jpg').convert('RGB')
                contact.thumbnail((640,900))
                x,y=(slot%2)*640,(slot//2)*940
                target.paste(contact,(x,y+40))
                ImageDraw.Draw(target).text((x+4,y+3),f'{n+1:02d} {sid}\n{r["task"]}',fill='black')
                media=meta[sid]['media']
                for col,kind in enumerate(('rgb','whitebox')):
                    member=next(m['member'] for m in media if m['field']==f'reference[0].{kind}')
                    im=Image.open(io.BytesIO(z.read(member))).convert('RGB').resize((640,360))
                    # Four pairs per page, preserving a useful image scale.
                    pair=OUT/f'reference_{n+1:02d}_{kind}.jpg'
                    im.save(pair,quality=94)
                rgb=Image.open(OUT/f'reference_{n+1:02d}_rgb.jpg').resize((360,202))
                wb=Image.open(OUT/f'reference_{n+1:02d}_whitebox.jpg').resize((360,202))
                refs.paste(rgb,(0,slot*210)); refs.paste(wb,(360,slot*210))
                ImageDraw.Draw(refs).text((730,slot*210+10),f'{n+1:02d} {sid}\n{r["task"]}',fill='black')
            target.save(OUT/f'targets_{page:02d}.jpg',quality=94)
            refs.save(OUT/f'references_{page:02d}.jpg',quality=94)
    print(json.dumps({k:v for k,v in manifest.items() if k!='records'},indent=2))

if __name__=='__main__':
    main()
