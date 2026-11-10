"""Stop only this project's foreground uploader before changing upload scope."""
import os
from pathlib import Path
import signal

for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except (OSError,ProcessLookupError):continue
    if any(a.endswith(b'/scripts/upload_gta_hf.py') for a in args) and b'upload' in args:
        os.kill(int(p.name),signal.SIGTERM)
        print('stopped_uploader_pid',p.name,flush=True)
