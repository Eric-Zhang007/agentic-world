"""Start the long resumable transfer outside the interactive terminal session."""
import json
import os
from pathlib import Path
import subprocess
import sys

base=Path('/mnt/e/GTA_memory_unified_20261021')
env=dict(os.environ,HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1')
log=base/'hf_upload.log'
with log.open('ab',buffering=0) as stream:
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('upload_gta_hf.py')),'--phase','upload'],
                           stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True,env=env)
(base/'HF_UPLOAD_JOB.json').write_text(json.dumps({'pid':child.pid,'log':str(log),'repo':'Isla8899/GTA-memory-training-20261021'},indent=2))
print('started_background_pid',child.pid,flush=True)
