"""Collect verified proxy receipts and missing media through existing SSH access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from server_exec import connect


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--old-source', type=Path, required=True)
    p.add_argument('--remote', default='/mnt/hdd/gta_source_rebuild/original_proxies')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    done = {}
    for path in a.output.glob('**/local_verified.json'):
        record = json.loads(path.read_text())
        media = Path(record['proxy'])
        if media.is_file() and media.stat().st_size == record['bytes']:
            done[record['output_relative']] = record
    state = {'status': 'syncing', 'completed': len(done), 'downloaded_bytes': 0,
             'reused_bytes': 0, 'started_at': time.time()}
    last_save = 0.0

    def save(force=False):
        nonlocal last_save
        if not force and time.monotonic() - last_save < 5:
            return
        state['updated_at'] = time.time()
        temp = a.output / 'SYNC_STATUS.tmp'
        temp.write_text(json.dumps(state, indent=2))
        for retry in range(30):
            try:
                temp.replace(a.output / 'SYNC_STATUS.json')
                break
            except PermissionError:
                if retry == 29:
                    raise
                time.sleep(.1)
        last_save = time.monotonic()

    save(True)
    while True:
        client = None
        try:
            client = connect()
            with client.open_sftp() as sftp:
                while True:
                    with sftp.open(a.remote + '/PROXY_STATUS.json') as f:
                        remote_status = json.loads(f.read())
                    with sftp.open(a.remote + '/proxy_results.jsonl') as f:
                        lines = f.read().splitlines()
                    available = {}
                    for line in lines:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if row['status'] == 'PROXY_DECODE_VERIFIED':
                            available[row['output_relative']] = row
                    state['remote_completed'] = remote_status['completed']
                    state['remote_failed'] = remote_status['failed']
                    for key, row in available.items():
                        if key in done:
                            continue
                        state['active_file'] = key
                        state['active_bytes'] = 0
                        save()
                        root = a.output / key
                        root.mkdir(parents=True, exist_ok=True)
                        old = a.old_source / key / 'proxy.mp4'
                        local = root / 'proxy.mp4'
                        if old.is_file() and old.stat().st_size == row['bytes'] and sha(old) == row['sha256']:
                            local = old
                            state['reused_bytes'] += row['bytes']
                        else:
                            if not local.exists() or sha(local) != row['sha256']:
                                part = local.with_name('proxy.mp4.downloading')

                                def progress(current, total):
                                    state['active_bytes'] = current
                                    state['active_total_bytes'] = total
                                    save()

                                # Receipts may store paths relative to the renderer's cwd.
                                remote_video = a.remote.rstrip('/') + '/' + key + '/proxy.mp4'
                                sftp.get(remote_video, str(part), callback=progress)
                                if sha(part) != row['sha256']:
                                    raise ValueError('Downloaded proxy content differs: ' + key)
                                part.replace(local)
                                state['downloaded_bytes'] += row['bytes']
                        result = {**row, 'remote_proxy': row['proxy'], 'proxy': str(local),
                                  'local_hash_matches': True}
                        (root / 'local_verified.json').write_text(json.dumps(result, indent=2))
                        done[key] = result
                        state['completed'] = len(done)
                        state['status'] = 'syncing'
                        state.pop('last_connection_error', None)
                        save()
                    if remote_status['status'] in ('complete', 'needs_review'):
                        state['status'] = 'complete' if len(done) == remote_status['total'] else 'needs_review'
                        save(True)
                        return
                    state['status'] = 'waiting_for_verified_proxies'
                    save(True)
                    time.sleep(15)
                    state['status'] = 'syncing'
        except (OSError, EOFError) as exc:
            state['status'] = 'reconnecting'
            state['last_connection_error'] = str(exc)
            save(True)
            time.sleep(10)
        finally:
            if client:
                client.close()


if __name__ == '__main__':
    main()
