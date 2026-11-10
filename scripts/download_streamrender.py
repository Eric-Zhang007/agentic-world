"""Acquire only new renderer assets; reuse the server's existing H3 encoders."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import HfApi, snapshot_download


def main(root):
    root.mkdir(parents=True,exist_ok=True)
    repo = 'Efficient-Large-Model/StreamRender-H3'
    api = HfApi(endpoint=os.environ['HF_ENDPOINT'])
    info = api.model_info(repo)
    revision = info.sha
    selected = [s.rfilename for s in info.siblings if
                s.rfilename.startswith(('backbone/','causal_decoder/','tae/')) or
                s.rfilename in ('assets.json','README.md','download_assets.py','inventory.json')]
    (root/'download_selection.json').write_text(json.dumps({'repo':repo,'revision':revision,
        'endpoint':os.environ['HF_ENDPOINT'],'files':selected,
        'reused':'Existing upstream Qwen, audio VAE and tokenizer; verify compatibility separately'},indent=2))
    start=time.monotonic()
    print(json.dumps({'event':'download_start','files':selected}),flush=True)
    snapshot_download(repo,revision=revision,local_dir=root,allow_patterns=selected,max_workers=6)
    print(json.dumps({'event':'download_complete','seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
