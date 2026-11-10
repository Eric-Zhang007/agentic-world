"""Show exact archived RGB/whitebox historical references for flagged sources."""
import io
import json
from pathlib import Path
import zipfile
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]/'outputs/dataset_reaudit'


def main():
    output=ROOT/'reference_contacts'
    output.mkdir(parents=True,exist_ok=True)
    sources=[json.loads(line) for line in (ROOT/'metadata/source_clock_rates.jsonl').read_text().splitlines()]
    selected=[r['sample_ids'][0] for r in sources if r['flagged']]
    cards=[]
    with zipfile.ZipFile('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip') as bundle:
        for sid in selected:
            prefix='samples/'+sid+'/'
            sample=json.loads(bundle.read(prefix+'sample.json'))
            refs=sample['conditions'].get('reference_pairs',[])
            if not refs and sample['conditions'].get('reference_pair'):
                refs=[sample['conditions']['reference_pair']]
            if not refs:
                raise ValueError('No declared reference pairs: '+sid)
            for index,ref in enumerate(refs):
                card=Image.new('RGB',(1280,390),'white')
                draw=ImageDraw.Draw(card)
                draw.text((5,3),sid+f' reference {index}',fill='black')
                for col,key in enumerate(('rgb','whitebox')):
                    image=Image.open(io.BytesIO(bundle.read(prefix+ref[key]))).convert('RGB')
                    image.thumbnail((640,360))
                    card.paste(image,(col*640,30))
                    draw.text((col*640+5,16),'HISTORICAL '+key,fill='black')
                cards.append((sid,index,card,ref))
    manifest=[]
    for offset in range(0,len(cards),3):
        page=Image.new('RGB',(1280,390*min(3,len(cards)-offset)),'white')
        filename=f'references_{offset//3:02d}.jpg'
        for row,(sid,index,card,ref) in enumerate(cards[offset:offset+3]):
            page.paste(card,(0,row*390))
            manifest.append({'sample_id':sid,'reference_index':index,'declared_pair':ref,'page':filename,'row':row})
        page.save(output/filename,quality=92)
    (output/'index.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'references':len(cards),'pages':(len(cards)+2)//3}))


if __name__=='__main__':main()
