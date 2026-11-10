"""Retry completed, failed source-recovery passes without concurrent writers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--old-source', type=Path, required=True)
    parser.add_argument('--ossutil', required=True)
    args = parser.parse_args()
    recovery = args.root / 'recovery'
    passes = 0
    while True:
        try:
            state = json.loads((recovery / 'RECOVERY_STATUS.json').read_text())
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            time.sleep(30)
            continue
        if state['status'] == 'complete':
            print('RECOVERY_COMPLETE', flush=True)
            return
        if state['status'] != 'needs_review':
            time.sleep(30)
            continue
        if state.get('conflicts') or passes >= 3:
            raise RuntimeError('Recovery needs inspection: conflicts or three exhausted retry passes')
        passes += 1
        print('RETRY_PASS', passes, 'failed_files', state.get('failed'), flush=True)
        time.sleep(60)
        command = [sys.executable, '-u', str(Path(__file__).with_name('recover_gta555_source.py')),
                   '--source', str(args.root / 'source'), '--old-source', str(args.old_source),
                   '--inventory', str(recovery / 'oss_objects.jsonl'), '--output', str(recovery),
                   '--ossutil', args.ossutil, '--workers', '2', '--apply']
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
