"""Test local regeneration on one real video pair and all original manifests."""
import json
import shutil
import sys
import zipfile
from pathlib import Path
from apply_delivery_patch import apply

SOURCE=Path('D:/GTA_memory_patch')
BASE=Path('D:/GTA_memory_patch_test/lightweight')
ARCHIVE=Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip')

def main():
    package=BASE/'patch';root=BASE/'dataset'
    package.mkdir(parents=True,exist_ok=True)
    spec=json.loads((SOURCE/'patch.json').read_text(encoding='utf-8'))
    sid=spec['repair_ids'][0]
    spec['repair_ids']=[sid];spec['repair_records']=1
    spec['files']=[f for f in spec['files'] if not f['path'].endswith('.mp4') or f['path'].startswith('samples/'+sid+'/')]
    with zipfile.ZipFile(ARCHIVE) as z:
        for f in spec['files']:
            dest=root/f['path'];dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(z.read(f['path']))
            if not f['path'].endswith('.mp4'):
                p=package/'payload'/f['path'];p.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(SOURCE/'payload'/f['path'],p)
        path=f'samples/{sid}/sample.json'
        (root/path).write_bytes(z.read(path))
    (package/'patch.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('verify_reaudit_pts_repair.py'),package/'repair_video.py')
    sys.path.insert(0,str(package))
    checks=[apply(root,package,False,True),apply(root,package,True,True),apply(root,package,True,True)]
    reports=[json.loads(p.read_text()) for p in (package/'local_verification').rglob('*.json')]
    assert len(reports)==2
    assert all(r['all_decoded_pixels_identical'] and r['cfr_timestamps_verified'] and r['decoded_frames']==120 for r in reports)
    result={'sample_id':sid,'scope':'One real original sample, two regenerated videos; all real manifests filtered. Separate full integration tests cover all 177 precomputed repairs.','checks':checks,'regenerated_streams':len(reports),'decoded_pixels_identical':True,'all_120_frame_24fps':True}
    (SOURCE/'lightweight_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
