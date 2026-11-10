"""Make a bounded RGB/proxy survey from the existing valid-run inventory."""
import json
from pathlib import Path
import cv2
import numpy as np

def main():
    out=Path('E:/GTA_memory_prepared/survey_20261015')
    out.mkdir(exist_ok=True)
    records=[json.loads(s) for s in Path('E:/GTA_memory_prepared/index_all_available/segments.jsonl').read_text(encoding='utf-8').splitlines()]
    chosen=[]
    groups={}
    for record in records:
        if record.get('camera_metadata_valid_runs'):
            groups.setdefault((record['episode_root'],record['agent_id']),[]).append(record)
    for key, group in sorted(groups.items())[:8]:
        for record in [group[0],group[len(group)//2]]:
            run=next((r for r in record['camera_metadata_valid_runs'] if r['frames']>=240),None)
            if run:
                chosen.append((record,run))
    cards=[]; results=[]
    for index,(record,run) in enumerate(chosen):
        path=Path(record['manifest']); manifest=json.loads(path.read_text(encoding='utf-8')); root=Path(record['episode_root'])
        frames=[]
        with (path.parent/'frames.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row['proxy_frame_index']==run['start_frame']+120:
                    frame=row;break
        for key,idx in [('rgb_video','rgb_decode_index'),('proxy_video','proxy_frame_index')]:
            cap=cv2.VideoCapture(str(root/manifest[key]));cap.set(cv2.CAP_PROP_POS_FRAMES,frame[idx]);ok,img=cap.read();cap.release()
            if not ok:break
            frames.append(cv2.resize(img,(640,360)))
        if len(frames)!=2:continue
        card=np.concatenate(frames,axis=1)
        label=f'{index}: {root.name[:20]} {record["agent_id"]} {manifest["segment"]} frame {frame["proxy_frame_index"]}'
        cv2.rectangle(card,(0,0),(1280,30),(0,0,0),-1)
        cv2.putText(card,label,(8,22),0,.55,(255,255,255),1)
        cv2.imwrite(str(out/f'{index:02d}.jpg'),card);cards.append(card)
        results.append({'survey_index':index,'manifest':str(path),'frame':frame['proxy_frame_index'],'valid_run':run})
    for start in range(0,len(cards),4):
        cv2.imwrite(str(out/f'sheet_{start//4}.jpg'),np.concatenate(cards[start:start+4],axis=0))
    (out/'index.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(len(results),out)

if __name__=='__main__':main()
