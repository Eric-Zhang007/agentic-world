"""Deploy only Pi harness files through server_exec.connect(), then check startup."""
import argparse
import ast
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT.parent))
from server_exec import connect

REMOTE = '/home/star/agentic_world'
DEST = REMOTE + '/harness/pi'
NODE = '/home/star/.nvm/versions/node/v22.23.2/bin'
PYTHON = REMOTE + '/.venv/bin/python'
FILES = ('package.json', 'package-lock.json', 'run.py', 'world.ts',
         'world_tools.py', 'world_instructions.md', 'README.md',
         'deploy.py', 'deploy_probe.mjs', 'test_world_tools.py', 'test_review_images.mjs')


def dependencies():
    pending = [HERE / 'world_tools.py', ROOT / 'whitebox/godot/render.py']
    found = set()
    while pending:
        path = pending.pop()
        if path in found or not path.is_file():
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [x.name for x in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                candidate = ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if candidate.is_file():
                    pending.append(candidate)
                    for parent in candidate.parents:
                        if parent == ROOT:
                            break
                        if (parent / '__init__.py').is_file():
                            pending.append(parent / '__init__.py')
    found.discard(HERE / 'world_tools.py')
    found.add(ROOT / 'skills/render-rollout/SKILL.md')
    found.update((ROOT / 'whitebox/godot').glob('*.gd'))
    found.update((ROOT / 'whitebox/godot').glob('project.godot'))
    return sorted(found)


def command(client, script):
    _, stdout, _ = client.exec_command(script)
    stdout.channel.set_combine_stderr(True)
    while True:
        data = stdout.channel.recv(32768)
        if not data:
            break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    return stdout.channel.recv_exit_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    client = connect()
    try:
        with client.open_sftp() as sftp:
            drift = []
            for path in dependencies():
                relative = path.relative_to(ROOT).as_posix()
                try:
                    with sftp.open(REMOTE + '/' + relative, 'rb') as remote:
                        identical = remote.read().replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n')
                    status = 'MATCH' if identical else 'DIFFERENT'
                except FileNotFoundError:
                    status = 'MISSING'
                drift.append({'path': relative, 'status': status})
            print(json.dumps({'dependency_comparison': drift}, indent=2), flush=True)
            if not args.check_only:
                for directory in (REMOTE + '/harness', DEST):
                    try:
                        sftp.stat(directory)
                    except FileNotFoundError:
                        sftp.mkdir(directory)
                for name in FILES:
                    sftp.put(str(HERE / name), DEST + '/' + name)
                    print('Uploaded ' + name, flush=True)
        prefix = 'export PATH=' + shlex.quote(NODE) + ':"$PATH"; cd ' + shlex.quote(DEST) + ' && '
        if not args.check_only:
            code = command(client, prefix + 'npm ci --no-audit --no-fund --cache ' + shlex.quote(DEST + '/.npm-cache'))
            if code:
                return code
        imports = """import importlib, sys
sys.path.insert(0, '/home/star/agentic_world')
failed = []
for name in ['yaml', 'numpy', 'av', 'PIL', 'requests', 'state.db',
             'whitebox.world', 'whitebox.motion', 'whitebox.video_reconcile',
             'perception.frame_sampler', 'renderer.comfy_h3',
             'scripts.reconcile_whitebox_video']:
    try:
        module = importlib.import_module(name)
        print(name, 'OK', getattr(module, '__version__', ''))
    except Exception as error:
        failed.append(name)
        print(name, type(error).__name__)
sys.exit(bool(failed))
"""
        imports_code = command(client, prefix + shlex.quote(PYTHON) +
                               ' -B -c ' + shlex.quote(imports))
        # run.py's model-list check writes its normal outputs/pi_harness/models.json,
        # which contains an environment placeholder, never the configured key.
        check = command(client, prefix + shlex.quote(PYTHON) + ' run.py --config ' +
                        shlex.quote(REMOTE + '/configs/default.yaml') + ' --session deploy-check --check')
        probe = command(client, prefix + 'WORLD_PYTHON=' + shlex.quote(PYTHON) +
                        ' PI_OFFLINE=1 PI_TELEMETRY=0 node deploy_probe.mjs')
        print(json.dumps({'pi_check_exit': check, 'extension_probe_exit': probe,
                          'dependency_import_exit': imports_code,
                          'dependency_drift': [x for x in drift if x['status'] != 'MATCH']}), flush=True)
        return check or probe or imports_code
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
