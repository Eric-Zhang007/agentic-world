"""Apply the GTA delivery correction to an extracted dataset, with backups."""
import argparse
import json
import shutil
from pathlib import Path

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def apply(root, package, write=False, rebuild_media=False):
    root=root.resolve(); package=package.resolve()
    spec=json.loads((package/'patch.json').read_text(encoding='utf-8'))
    marker=root/'GTA_MEMORY_PATCH_APPLIED.json'
    already=marker.exists()
    backup=root/'patch_backup'
    for f in spec['files']:
        relative=Path(f['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid relative path')
        src=package/'payload'/relative
        missing_rebuild=rebuild_media and relative.suffix=='.mp4' and not src.exists()
        if not missing_rebuild and (not src.is_file() or src.stat().st_size!=f['bytes']):
            raise ValueError(f'Missing or incomplete payload: {relative}')
    for name,original in spec['original_manifest_records'].items():
        corrected=rows(package/'payload'/name)
        expected=corrected if already else original
        actual=rows(root/name)
        if actual!=expected:
            if already or actual!=corrected or not (backup/name).exists() or rows(backup/name)!=original:
                raise ValueError(f'Dataset manifest does not match this patch: {name}')
    for sid in spec['repair_ids']:
        sample=json.loads((root/'samples'/sid/'sample.json').read_text(encoding='utf-8'))
        if sample.get('fps')!=24 or sample.get('frame_count')!=120:
            raise ValueError(f'Unexpected video contract: {sid}')
    if already:
        for f in spec['files']:
            if (root/f['path']).stat().st_size!=f['bytes']:
                raise ValueError('Applied file size differs: '+f['path'])
        return {'status':'already_applied','records':spec['remaining']}
    if not write:
        return {'status':'check_passed','repairs':spec['repair_records'],'exclusions':spec['exclude_records'],'records':spec['remaining'],'local_media_rebuild':rebuild_media}
    if rebuild_media:
        from repair_video import candidate, compare
        for f in spec['files']:
            dest=package/'payload'/f['path']
            if dest.exists():
                continue
            original=root/f['path']
            data=original.read_bytes()
            temp=dest.with_name(dest.stem+'.building.mp4')
            dest.parent.mkdir(parents=True,exist_ok=True)
            candidate(data,temp,24)
            checked=compare(data,temp,120,24)
            temp.replace(dest)
            f['bytes']=dest.stat().st_size
            checked.pop('timestamps')
            verification=package/'local_verification'/f['path']
            verification=verification.with_suffix('.json')
            verification.parent.mkdir(parents=True,exist_ok=True)
            verification.write_text(json.dumps(checked,indent=2))
            # Persist completed work for restart, including codec-dependent file sizes.
            (package/'patch.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
            print('Repaired and pixel-verified: '+f['path'],flush=True)
    # Complete all backups before any replacement, allowing an interrupted copy to resume.
    for f in spec['files']:
        dest=root/f['path']; saved=backup/f['path']
        if not saved.exists():
            saved.parent.mkdir(parents=True,exist_ok=True)
            temp_saved=saved.with_name(saved.name+'.backup_tmp')
            shutil.copy2(dest,temp_saved)
            temp_saved.replace(saved)
    for f in spec['files']:
        dest=root/f['path']; temp=dest.with_name(dest.name+'.patch_tmp')
        shutil.copy2(package/'payload'/f['path'],temp)
        temp.replace(dest)
    for name in spec['original_manifest_records']:
        if rows(root/name)!=rows(package/'payload'/name):
            raise ValueError('Manifest verification failed')
    marker.write_text(json.dumps({'records':spec['remaining'],'repaired':spec['repair_records'],'excluded':spec['exclude_records']},indent=2))
    return {'status':'applied','records':spec['remaining']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--rebuild-media',action='store_true',help='Recreate omitted videos locally with av/numpy; compare every decoded pixel')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true');g.add_argument('--apply',action='store_true')
    a=p.parse_args()
    print(json.dumps(apply(a.root,Path(__file__).resolve().parent,a.apply,a.rebuild_media),ensure_ascii=False,indent=2))
