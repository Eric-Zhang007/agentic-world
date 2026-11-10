"""Print account identity and credential availability without credential values."""
import importlib.util
import json
import os
from pathlib import Path
import urllib.request

print('home',str(Path.home()),flush=True)
print('hub_installed',bool(importlib.util.find_spec('huggingface_hub')),flush=True)
token=os.environ.get('HF_TOKEN')
for path in [Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'token',Path.home()/'.huggingface/token']:
    if path.is_file():
        print('credential_file_exists',str(path),flush=True)
        if not token:token=path.read_text().strip()
print('credential_available',bool(token),flush=True)
if token:
    req=urllib.request.Request('https://huggingface.co/api/whoami-v2',headers={'Authorization':'Bearer '+token})
    try:
        with urllib.request.urlopen(req,timeout=30) as response:data=json.load(response)
        print(json.dumps({'name':data.get('name'),'orgs':[o.get('name') for o in data.get('orgs',[])],'auth_type':data.get('auth',{}).get('type')}),flush=True)
    except Exception as exc:print('identity_request_failed',type(exc).__name__,str(exc),flush=True)
