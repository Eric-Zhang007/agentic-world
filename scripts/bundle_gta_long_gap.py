"""Package reviewed longer-gap diagnostics and separately labeled state audit."""
import json
import zipfile
from pathlib import Path
from scripts.prepare_gta_object_memory import read
from memory.gta_validation import validate_reference_identity, camera_exclusion


def main():
    root=Path('E:/GTA_memory_prepared')
    output=root/'object_memory_long_gap_state_20261016.zip'
    if output.exists():raise ValueError('Use a new archive name')
    names=['person_long_gap_a000','person_long_gap_a001']
    index=[]
    for name in names:
        p=root/name; package=read(p/'package.json');review=read(p/'visual_review.json');obs=read(p/'observations.json')
        validate_reference_identity(obs,review['dataset_entity_id'])
        if any(camera_exclusion(o['frame_metadata']) for o in obs):raise ValueError('Invalid camera')
        index.append({'sample_id':name,'sample_root':name,'manifest':name+'/samples.jsonl',
            'reference_age_sec':(package['target_start_time_ns']-package['reference_time_ns'])/1e9,
            'target_frames':package['target_frames'],'training_ready':False,
            'eligibility':'appearance_diagnostic_only' if name.endswith('a001') else 'coarse_appearance_pilot'})
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('batch_index.json',json.dumps(index,indent=2))
        archive.write(root/'BATCH_LONG_GAP_20261016.md','README.md')
        for name in names+['vehicle_state_audit']:
            for file in sorted((root/name).rglob('*')):
                if file.is_file():archive.write(file,file.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():raise ValueError('Integrity error')
        for entry in index:
            sample=json.loads(archive.read(entry['manifest']))
            for key in ['target_whitebox','prompt']:
                assert entry['sample_root']+'/'+sample['conditions'][key] in archive.namelist()
            assert entry['sample_root']+'/'+sample['supervision']['target_rgb'] in archive.namelist()
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'samples':index}))


if __name__=='__main__':main()
