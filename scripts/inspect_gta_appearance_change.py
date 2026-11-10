"""Audit one real appearance boundary without assuming entity continuity."""
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.prepare_gta_object_memory import read, write, selected_records


def main():
    root=Path('E:/GTA_555/0921')
    seg=root/'derived/segments/agent_000/segment_00075'
    out=Path('E:/GTA_memory_prepared/appearance_change_0921_s75')
    out.mkdir(exist_ok=True)
    manifest=read(seg/'manifest.json')
    frames=[json.loads(x) for x in (seg/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[frames[i] for i in [280,300,319,320,340,380]]
    keys={f['client_sample_seq'] for f in selected}
    cameras=selected_records(root/'agents/agent_000/camera.parquet',keys,'client_sample_seq',selected[0]['client_clock_domain'])
    world=selected_records(root/'server/world_state.parquet',{f['world_capture_tick'] for f in selected},'sample_seq')
    rows=[];cards=[]
    for f in selected:
        camera=cameras[f['client_sample_seq']]
        subject=camera['subject_observation']
        actors=[a for a in world[f['world_capture_tick']]['agents'] if a.get('agent_id')=='agent_000']
        rows.append({'frame':f['proxy_frame_index'],'time_ns':f['server_time_estimate_ns'],
            'appearance_id':f.get('appearance_id'),'character_id':f.get('character_id'),
            'camera_interpolation':f['camera_interpolation'],'subject':subject,'world_actor':actors})
        images=[]
        for key,index in [('rgb_video','rgb_decode_index'),('proxy_video','proxy_frame_index')]:
            cap=cv2.VideoCapture(str(root/manifest[key]));cap.set(cv2.CAP_PROP_POS_FRAMES,f[index]);ok,image=cap.read();cap.release();assert ok
            images.append(cv2.resize(image,(640,360)))
        card=np.concatenate(images,axis=1)
        cv2.putText(card,str(f['proxy_frame_index']),(10,25),0,.8,(0,255,255),2);cards.append(card)
    cv2.imwrite(str(out/'boundary.jpg'),np.concatenate(cards,axis=0))
    write(out/'records.json',rows)
    print([(r['frame'],r['character_id'],[(a['dataset_entity_id'],a.get('model_hash')) for a in r['world_actor']]) for r in rows])


if __name__=='__main__':main()
