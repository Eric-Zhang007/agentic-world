"""Collect sparse camera/world evidence for a real vehicle state transition."""
import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from scripts.prepare_gta_object_memory import read, write, selected_records


def main():
    root=Path('E:/GTA_555/0921')
    segment=root/'derived/segments/agent_001/segment_00000'
    out=Path('E:/GTA_memory_prepared/vehicle_state_audit')
    out.mkdir(exist_ok=True)
    manifest=read(segment/'manifest.json')
    rows=[json.loads(s) for s in (segment/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    frames=[rows[i] for i in [720,780,840,900,960,1020,1080,1200]]
    cameras=selected_records(root/'agents/agent_001/camera.parquet',{f['client_sample_seq'] for f in frames},'client_sample_seq',frames[0]['client_clock_domain'])
    worlds=selected_records(root/'server/world_state.parquet',{f['world_capture_tick'] for f in frames},'sample_seq')
    result=[];cards=[]
    oid='3574744230133772'
    for f in frames:
        c=cameras[f['client_sample_seq']];w=worlds[f['world_capture_tick']]
        result.append({'frame':f['proxy_frame_index'],'server_time_ns':f['server_time_estimate_ns'],
            'camera_interpolation':f.get('camera_interpolation'),
            'client_vehicle':[e for e in c['scene_observation']['entities'] if str(e.get('dataset_entity_id'))==oid],
            'client_actors':[e for e in c['scene_observation']['entities'] if e.get('is_player')],
            'world_vehicle':[e for e in w.get('vehicles',[]) if str(e.get('dataset_entity_id'))==oid],
            'world_actors':w.get('agents',[])})
        images=[]
        for key,idx in [('rgb_video','rgb_decode_index'),('proxy_video','proxy_frame_index')]:
            cap=cv2.VideoCapture(str(root/manifest[key]));cap.set(1,f[idx]);ok,img=cap.read();cap.release();assert ok
            cv2.imwrite(str(out/f'{f["proxy_frame_index"]}_{key}.jpg'),img)
            images.append(cv2.resize(img,(640,360)))
        card=np.concatenate(images,axis=1);cv2.putText(card,str(f['proxy_frame_index']),(8,25),0,.8,(0,255,255),2);cards.append(card)
    for start in [0,4]:cv2.imwrite(str(out/f'sheet_{start}.jpg'),np.concatenate(cards[start:start+4],axis=0))
    events=[]
    for batch in pq.ParquetFile(root/'server/events.parquet').iter_batches(batch_size=512):
        for row in batch.to_pylist():
            e=json.loads(row['payload_json']);t=e.get('server_monotonic_ns')
            if t is not None and result[0]['server_time_ns'] <= t <= result[-1]['server_time_ns']:
                events.append(e)
    write(out/'evidence.json',{'object_id':oid,'episode_id':manifest['episode_id'],'frames':result,'events_in_window':events})
    print([(r['frame'],[(e.get('doors'),e.get('driver_entity_id')) for e in r['world_vehicle']]) for r in result])


if __name__=='__main__':main()
