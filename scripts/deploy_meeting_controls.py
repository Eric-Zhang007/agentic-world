"""Deploy the small interactive entrypoint without changing any running world."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import server_exec

root = Path(__file__).resolve().parents[1]
files = ['scripts/serve_controls.py', 'scripts/run_interactive_rollout.py',
         'harness/controls/index.html', 'harness/controls/app.js',
         'scripts/test_world_bound_controls.py', 'scripts/test_video_reconcile.py']
client = server_exec.connect()
sftp = client.open_sftp()
backup = '/home/star/agentic_world/outputs/meeting_demo/ui_backup_' + str(int(time.time()))
sftp.mkdir(backup)
for name in files:
    target = '/home/star/agentic_world/' + name
    try:
        with sftp.open(target, 'rb') as f:
            data = f.read()
        with sftp.open(backup + '/' + name.replace('/', '__'), 'wb') as f:
            f.write(data)
    except FileNotFoundError:
        pass
    sftp.put(str(root / name), target)
sftp.close()
_, out, err = client.exec_command('cd /home/star/agentic_world && .venv/bin/python -m unittest scripts.test_world_bound_controls scripts.test_video_reconcile')
print(out.read().decode(), err.read().decode())
code = out.channel.recv_exit_status()
client.close()
raise SystemExit(code)
