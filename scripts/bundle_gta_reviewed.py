"""Bundle already reviewed data pilots, with explicit per-sample path roots."""
import json
import zipfile
from pathlib import Path


def main():
    root=Path('E:/GTA_memory_prepared')
    names=['pilot_0921_a000_s00001_short','person_a000_reviewed','person_a001_reviewed']
    output=root/'object_memory_reviewed_20261016.zip'
    index=[]
    if output.exists():raise ValueError('Archive already exists')
    for name in names:
        package=json.loads((root/name/'package.json').read_text(encoding='utf-8'))
        if package['status']!='VISUALLY_REVIEWED_OBJECT_MEMORY_PILOT':raise ValueError(name)
        index.append({'sample_id':name,'sample_root':name,'manifest':name+'/samples.jsonl',
                      'target_frames':package['target_frames'],'training_ready':False})
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr('batch_index.json',json.dumps(index,indent=2))
        archive.write(root/'BATCH_20261016.md','README.md')
        archive.write(root/'review_20261016.json','review_20261016.json')
        for name in names:
            for file in sorted((root/name).rglob('*')):
                if file.is_file():archive.write(file,file.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise ValueError('Archive integrity failed')
        for item in index:
            sample=json.loads(archive.read(item['manifest']).decode('utf-8'))
            paths=[sample['conditions']['prompt'],sample['conditions']['target_whitebox'],
                   sample['supervision']['target_rgb'],sample['frame_state_index'],sample['camera_records'],sample['world_records']]
            for pair in sample['conditions']['reference_pairs']:paths.extend(pair.values())
            for path in paths:
                if item['sample_root']+'/'+path not in archive.namelist():raise ValueError(path)
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'samples':len(index),
                      'target_frames':sum(r['target_frames'] for r in index)}))


if __name__=='__main__':main()
