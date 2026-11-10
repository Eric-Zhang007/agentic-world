"""Upload the reviewed delivery using the existing WSL HF session."""
import argparse
import json
import os
from pathlib import Path
import threading
import time
from huggingface_hub import HfApi

p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['prepare','upload','public'],required=True)
args=p.parse_args()
api=HfApi()
repo='Isla8899/GTA-memory-training-20261021'
base=Path('/mnt/e/GTA_memory_unified_20261021')
folder=base/'hf_upload'
folder.mkdir(exist_ok=True)
if args.phase=='public':
    api.update_repo_settings(repo_id=repo,repo_type='dataset',private=False)
    info=api.repo_info(repo,repo_type='dataset')
    assert not info.private
    print(json.dumps({'repo':repo,'private':info.private}),flush=True)
elif args.phase=='prepare':
    url=api.create_repo(repo_id=repo,repo_type='dataset',private=True,exist_ok=True)
    info=api.repo_info(repo,repo_type='dataset')
    assert info.private, 'Expected private repository'
    print(json.dumps({'repo':repo,'private':info.private,'url':str(url)}),flush=True)
else:
    for path,name in [(Path('/mnt/c/Users/zjc/omgd/agentic_world/docs/gta_unified_delivery/HF_README.md'),'README.md'),
                      (base/'DELIVERY.json','DELIVERY.json')]:
        print('UPLOADING',name,flush=True)
        api.upload_file(path_or_fileobj=str(path),path_in_repo=name,repo_id=repo,repo_type='dataset')
    remote=api.list_repo_files(repo_id=repo,repo_type='dataset')
    for name in ('GTA_memory_acceptance_under1GB.zip','ACCEPTANCE_REPORT.json'):
        if name in remote:api.delete_file(path_in_repo=name,repo_id=repo,repo_type='dataset',commit_message='Keep acceptance subset local as requested')
    full=base/'GTA_memory_training_unified.zip'
    status=base/'HF_UPLOAD_STATUS.json'
    finished=threading.Event()
    def heartbeat():
        while not finished.is_set():
            status.write_text(json.dumps({'status':'UPLOADING','pid':os.getpid(),'repo':repo,'updated_unix':time.time(),'file':full.name}))
            finished.wait(30)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        api.upload_large_folder(repo_id=repo,repo_type='dataset',folder_path=str(base),allow_patterns=[full.name],num_workers=2,print_report_every=30)
    finally:
        finished.set()
    files={f.rfilename:f.size for f in api.repo_info(repo,repo_type='dataset',files_metadata=True).siblings}
    assert files.get(full.name)==full.stat().st_size
    result={'status':'UPLOADED_AND_REMOTE_SIZE_VERIFIED','repo':repo,'private':api.repo_info(repo,repo_type='dataset').private,'files':files}
    (base/'HF_UPLOAD_RESULT.json').write_text(json.dumps(result,indent=2))
    status.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
