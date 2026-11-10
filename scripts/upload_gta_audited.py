"""Upload one verified unified delivery to a named public dataset subdirectory."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery',type=Path,required=True)
    p.add_argument('--repo',default='Isla8899/GTA-memory-training-20261021')
    p.add_argument('--remote-directory',default='reviewed_local')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    api=HfApi()
    info=api.repo_info(a.repo,repo_type='dataset')
    if info.private:raise ValueError('Expected existing public repository')
    if a.check_only:
        who=api.whoami()
        print(json.dumps({'repo':a.repo,'public':True,'authenticated_user':who['name']}));return
    delivery=json.loads((a.delivery/'DELIVERY.json').read_text())
    verification=json.loads((a.delivery/'package_metadata/VERIFICATION.json').read_text())
    archive=a.delivery/delivery['archive']
    if delivery['status']!='PACKAGED_AND_VERIFIED' or verification['passed'] is not True:
        raise ValueError('Package has not passed verification')
    if archive.stat().st_size!=delivery['bytes']:raise ValueError('Archive size changed')
    prefix=a.remote_directory.strip('/')
    if not prefix or '..' in prefix.split('/'):raise ValueError('Use a new explicit remote directory')
    remote=f'{prefix}/{archive.name}'
    existing=api.list_repo_files(a.repo,repo_type='dataset')
    if remote in existing:raise ValueError('Existing remote delivery will not be overwritten')
    (a.delivery/'HF_UPLOAD_STATUS.json').write_text(json.dumps({'status':'UPLOADING','repo':a.repo,'remote':remote}))
    api.upload_file(path_or_fileobj=str(archive),path_in_repo=remote,repo_id=a.repo,repo_type='dataset',
                    commit_message='Add audited local scene and object memory training package')
    for path,name in [(a.delivery/'DELIVERY.json','DELIVERY.json'),
                      (a.delivery/'package_metadata/README.md','README.md'),
                      (a.delivery/'package_metadata/VERIFICATION.json','VERIFICATION.json'),
                      (a.delivery/'package_metadata/DATASET_REPORT.json','DATASET_REPORT.json')]:
        api.upload_file(path_or_fileobj=str(path),path_in_repo=f'{prefix}/{name}',repo_id=a.repo,repo_type='dataset')
    sizes={f.rfilename:f.size for f in api.repo_info(a.repo,repo_type='dataset',files_metadata=True).siblings}
    if sizes.get(remote)!=archive.stat().st_size:raise ValueError('Remote size verification failed')
    report={'status':'UPLOADED_AND_REMOTE_SIZE_VERIFIED','repo':a.repo,'remote':remote,'bytes':sizes[remote]}
    (a.delivery/'HF_UPLOAD_STATUS.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
