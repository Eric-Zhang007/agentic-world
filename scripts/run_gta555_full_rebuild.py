"""Execute frozen reproduction jobs; rendering is not training qualification."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('workspace', 'source_root', 'outputs'):
        if not isinstance(config.get(key), str):
            raise ValueError(f'Missing path setting: {key}')
    return config


def verify_render(destination, job):
    import av
    root = Path(destination) / 'render'
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Renderer did not complete')
    info = manifest['agents'][job['agent']]
    count = 0
    with av.open(str(root / info['video'])) as video:
        for frame in video.decode(video=0):
            if not frame.width or not frame.height:
                raise ValueError('Empty decoded frame')
            count += 1
    with (root / info['frame_index']).open() as stream:
        rows = sum(bool(line.strip()) for line in stream)
    if count != job['expected_frames'] or rows != count or info['frames'] != count:
        raise ValueError(f'Frame count mismatch decoded={count} metadata={rows} expected={job["expected_frames"]}')
    usable = info['statuses'].get('estimated_alignment', 0)
    if usable != job['usable_frames']:
        raise ValueError(f'Usable frame mismatch: {usable} != {job["usable_frames"]}')
    return {'decoded_frames': count, 'usable_frames': usable,
            'proxy': str(root / info['video']), 'frames': str(root / info['frame_index'])}


def run_job(source, output, job):
    key = '_'.join(job[k] for k in ('episode', 'agent', 'segment'))
    parent = output / key
    parent.mkdir(parents=True, exist_ok=True)
    receipt = parent / 'verified.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        return {**previous, **verify_render(previous['destination'], job), 'resumed': True}
    attempt = parent / ('attempt_' + str(time.time_ns()))
    command = [sys.executable, str(Path(__file__).with_name('gta555_render_entry.py')), '--source', str(source),
               '--episode', job['episode'], '--agent', job['agent'],
               '--segment', job['segment'], '--destination', str(attempt)]
    started = time.time()
    with (parent / (attempt.name + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = {**job, **verify_render(attempt, job), 'destination': str(attempt),
              'render_status': 'decoded_verified', 'training_status': 'PENDING_QUALIFICATION',
              'elapsed_sec': time.time() - started}
    receipt.write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/gta555_full_rebuild.yaml'))
    p.add_argument('--jobs', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cfg = load_config(a.config)
    source = Path(cfg['source_root']).resolve()
    output = Path(cfg['outputs']).resolve()
    if output == source or source in output.parents:
        raise ValueError('Outputs must be outside immutable source')
    jobs = [j for j in json.loads(a.jobs.read_text()) if j['ready']]
    if a.limit is not None:
        jobs = jobs[:a.limit]
    if a.dry_run:
        print(json.dumps({'source': str(source), 'jobs': len(jobs), 'workers': cfg['render']['workers']}))
        return
    if os.name == 'nt':
        raise SystemExit('Frozen Mesa libraries require Linux; use the server render environment')
    if not jobs:
        raise SystemExit('No source-ready jobs; inspect readiness.json and missing_files.jsonl')
    output.mkdir(parents=True, exist_ok=True)
    state = {'status': 'rendering', 'total': len(jobs), 'completed': 0, 'failed': 0,
             'started_at': time.time(), 'downstream': 'camera qualification and training export pending'}

    def save():
        state['updated_at'] = time.time()
        temporary = output / 'PIPELINE_STATUS.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output / 'PIPELINE_STATUS.json')

    save()
    with ThreadPoolExecutor(max_workers=int(cfg['render']['workers'])) as pool, (output / 'render_results.jsonl').open('a') as ledger:
        pending = {pool.submit(run_job, source, output, job): job for job in jobs}
        while pending:
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    result = future.result()
                    state['completed'] += 1
                except Exception as exc:
                    result = {**job, 'render_status': 'failed', 'error': str(exc)}
                    state['failed'] += 1
                ledger.write(json.dumps(result) + '\n')
                ledger.flush()
            active = []
            for future, job in pending.items():
                if not future.running():
                    continue
                key = '_'.join(job[k] for k in ('episode', 'agent', 'segment'))
                attempts = sorted((output / key).glob('attempt_*/render/' + job['agent'] + '/frames.jsonl'))
                count = 0
                if attempts:
                    with attempts[-1].open() as stream:
                        count = sum(bool(line.strip()) for line in stream)
                active.append({'job': key, 'frame_rows_written': count, 'expected_frames': job['expected_frames']})
            state['active_jobs'] = active
            save()
            if done:
                print(json.dumps(state), flush=True)
    state['status'] = 'rendered' if not state['failed'] else 'render_failures'
    save()
    if state['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
