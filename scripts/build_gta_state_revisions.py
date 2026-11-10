"""Export append-only observed states with explicit uncertainty and source times."""
import json
from pathlib import Path
from scripts.prepare_gta_object_memory import read, write


def main():
    root=Path('E:/GTA_memory_prepared/vehicle_state_audit')
    evidence=read(root/'evidence.json')
    review=read(Path(__file__).resolve().parents[1]/'configs/gta_vehicle_state_review.json')
    assert evidence['episode_id']==review['episode_id'] and evidence['object_id']==review['object_id']
    frames={r['frame']:r for r in evidence['frames']}
    observations=[]
    for annotation in review['observations']:
        record=frames[annotation['frame']]
        vehicle=record['world_vehicle']
        assert len(vehicle)==1 and str(vehicle[0]['dataset_entity_id'])==review['object_id']
        observations.append({'observation_id':f'vehicle_state_f{annotation["frame"]}',
            'object_id':review['episode_id']+':entity:'+review['object_id'],
            'observed_at_ns':record['server_time_ns'], 'frame':annotation['frame'],
            'rgb':str(annotation['frame'])+'_rgb_video.jpg',
            'whitebox':str(annotation['frame'])+'_proxy_video.jpg',
            'source_world_door_state':vehicle[0].get('doors'),
            'source_world_driver_id':vehicle[0].get('driver_entity_id'),
            'visual_observation':annotation,
            'provenance':'manual_rgb_review', 'valid_until_ns':None,
            'validity':'sampled_instant_only; no inferred persistence interval',
            'supersedes':observations[-1]['observation_id'] if observations else None})
    assert all(a['observed_at_ns']<b['observed_at_ns'] for a,b in zip(observations,observations[1:]))
    write(root/'state_observations.json',{'review':review,'observations':observations,
        'authoritative_world_modified':False,'reference_use_edges':[],
        'note':'No consuming generation exists; do not invent actual reference usage. Supersedes orders newer observations without invalidating old historical truth.'})
    print('state observations:',len(observations))


if __name__=='__main__':main()
