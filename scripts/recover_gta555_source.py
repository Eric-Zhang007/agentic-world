"""Recover missing source files from old local data or authorized OSS.

Never replace existing source files. Size checks establish transport completeness,
not data equivalence; the bundle's hashes and spool indexes remain the final checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import shutil
import subprocess
import time


def destination_for(key):
    rel = Path(key.removeprefix('GTA_555/'))
    if not rel.parts or rel.parts[0] in ('_delivery', 'evidence'):
        return None
    if 'derived' in rel.parts and (rel.name.startswith('scene_') or
            rel.suffix in ('.png', '.log') or rel.name == 'proxy.mp4'):
        return None
    # These root documents are already supplied in the portable package.
    if len(rel.parts) == 1:
        return None
    return Path('data') / rel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--old-source', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ossutil', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    # The extractor and this recovery must never write the same tree concurrently.
    extraction = a.source.parent / 'EXTRACTION_STATUS.json'
    while extraction.exists() and json.loads(extraction.read_text())['status'].lower() not in ('complete', 'completed', 'extracted'):
        state = json.loads(extraction.read_text())
        if state['status'].lower() == 'failed':
            raise RuntimeError('Extraction failed; inspect its error before recovery')
        print('Waiting for source extraction', flush=True)
        time.sleep(10)
    inventory_state = a.inventory.parent / 'OSS_STATUS.json'
    if json.loads(inventory_state.read_text())['status'] != 'complete':
        raise RuntimeError('OSS inventory is still incomplete')
    plan = []
    conflicts = []
    total_existing = 0
    for line in a.inventory.open(encoding='utf-8'):
        row = json.loads(line)
        size = int(row['Size'])
        rel = destination_for(row['Key'])
        if row['Key'].endswith('/') or rel is None:
            continue
        dst = a.source / rel
        if dst.exists():
            if dst.stat().st_size == size:
                total_existing += 1
            else:
                conflicts.append({'path': str(rel), 'expected': size, 'actual': dst.stat().st_size})
            continue
        old = a.old_source / Path(row['Key'].removeprefix('GTA_555/'))
        local = old.is_file() and old.stat().st_size == size
        plan.append({'key': row['Key'], 'size': size, 'destination': str(dst),
                     'local': str(old) if local else None})
    # Finish small metadata first so individual render inputs become inspectable.
    plan.sort(key=lambda r: (r['size'] > 20_000_000, r['local'] is None, r['size']))
    (a.output / 'recovery_plan.json').write_text(json.dumps(plan, indent=2))
    (a.output / 'recovery_conflicts.json').write_text(json.dumps(conflicts, indent=2))
    state = {'status': 'planned', 'existing_size_matches': total_existing,
             'pending_files': len(plan), 'pending_bytes': sum(r['size'] for r in plan),
             'local_copy_bytes': sum(r['size'] for r in plan if r['local']),
             'download_bytes': sum(r['size'] for r in plan if not r['local']),
             'conflicts': len(conflicts), 'completed': 0, 'completed_bytes': 0,
             'failed': 0, 'started_at': time.time()}

    last_saved = 0.0

    def save(force=False):
        nonlocal last_saved
        if not force and time.monotonic() - last_saved < 5:
            return
        state['updated_at'] = time.time()
        tmp = a.output / 'RECOVERY_STATUS.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        for attempt in range(30):
            try:
                tmp.replace(a.output / 'RECOVERY_STATUS.json')
                break
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(.1)
        last_saved = time.monotonic()

    save()
    print(json.dumps(state), flush=True)
    if not a.apply:
        return

    def recover(row):
        dst = Path(row['destination'])
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + '.recovering')
        if dst.exists():
            raise RuntimeError(f'Destination appeared during recovery: {dst}')
        if row['local']:
            shutil.copyfile(row['local'], part)
        else:
            cmd = [a.ossutil, 'cp', 'oss://gta-data-lqr/' + row['key'], str(part),
                   '--force', '--region', 'cn-beijing', '--endpoint',
                   'https://oss-cn-beijing.aliyuncs.com', '--parallel', '2',
                   '--retry-times', '3', '--quiet']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr[-1500:] or result.stdout[-1500:])
        if part.stat().st_size != row['size']:
            raise RuntimeError(f'Wrong downloaded size: {dst}')
        part.rename(dst)
        return row

    state['status'] = 'recovering'
    save(True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool, (a.output / 'recovery_results.jsonl').open('a') as log:
        pending = {}
        rows = iter(plan)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < a.workers:
                row = next(rows, None)
                if row is None:
                    exhausted = True
                else:
                    pending[pool.submit(recover, row)] = row
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                row = pending.pop(future)
                try:
                    future.result()
                    state['completed'] += 1
                    state['completed_bytes'] += row['size']
                    result = {'key': row['key'], 'status': 'copied' if row['local'] else 'downloaded'}
                except Exception as exc:
                    state['failed'] += 1
                    result = {'key': row['key'], 'status': 'failed', 'error': str(exc)}
                log.write(json.dumps(result) + '\n')
                log.flush()
            state['active_files'] = [r['key'] for r in pending.values()]
            save()
    state['status'] = 'complete' if not state['failed'] and not conflicts else 'needs_review'
    save(True)


if __name__ == '__main__':
    main()
