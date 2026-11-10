"""Fetch original proxy videos, verify publication hashes, decode every frame."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time
import av
import oss2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((a.source / 'data/segments.json').read_text())['segments']
    auth = oss2.Auth(os.environ['OSS_ACCESS_KEY_ID'], os.environ['OSS_ACCESS_KEY_SECRET'])

    def process(job):
        relative = job['output_relative']
        root = a.output / relative
        root.mkdir(parents=True, exist_ok=True)
        validation = root / 'validation.json'
        video = root / 'proxy.mp4'
        receipt = root / 'verified.json'
        bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', 'gta-data-lqr')

        def fetch(name, path):
            temporary = path.with_name(path.name + '.download')
            bucket.get_object_to_file('GTA_555/' + relative + '/' + name, str(temporary))
            temporary.replace(path)

        if not validation.exists():
            local = a.source / 'data' / relative / 'validation.json'
            if local.exists():
                validation.write_bytes(local.read_bytes())
            else:
                fetch('validation.json', validation)
        if digest(validation) != job['validation_sha256']:
            raise ValueError('Validation differs from segment index')
        metadata = json.loads(validation.read_text())
        expected_hash = metadata['published_sha256']['proxy.mp4']
        if not video.exists():
            fetch('proxy.mp4', video)
        if digest(video) != expected_hash:
            raise ValueError('Proxy differs from publication hash')
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous.get('sha256') == expected_hash and previous.get('decoded_frames') == job['frames']:
                return previous
        count = 0
        last_pts = None
        with av.open(str(video)) as container:
            fps = str(container.streams.video[0].average_rate)
            for frame in container.decode(video=0):
                if frame.pts is None or (last_pts is not None and frame.pts <= last_pts):
                    raise ValueError('Missing or non-monotonic proxy PTS')
                last_pts = frame.pts
                count += 1
        if count != job['frames']:
            raise ValueError(f'Decoded {count} frames, expected {job["frames"]}')
        result = {**job, 'proxy': str(video), 'sha256': expected_hash,
                  'decoded_frames': count, 'fps': fps, 'bytes': video.stat().st_size,
                  'status': 'PROXY_DECODE_VERIFIED', 'training_status': 'PENDING_RGB_CAMERA_VISUAL_QA'}
        receipt.write_text(json.dumps(result, indent=2))
        return result

    state = {'status': 'verifying', 'total': len(jobs), 'completed': 0, 'failed': 0,
             'verified_bytes': 0, 'decoded_frames': 0, 'started_at': time.time()}

    def save():
        state['updated_at'] = time.time()
        tmp = a.output / 'PROXY_STATUS.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(a.output / 'PROXY_STATUS.json')

    save()
    with ThreadPoolExecutor(max_workers=a.workers) as pool, (a.output / 'proxy_results.jsonl').open('a') as ledger:
        pending = {}
        remaining = iter(jobs)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < a.workers and not exhausted:
                job = next(remaining, None)
                if job is None:
                    exhausted = True
                else:
                    pending[pool.submit(process, job)] = job
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    result = future.result()
                    state['completed'] += 1
                    state['verified_bytes'] += result['bytes']
                    state['decoded_frames'] += result['decoded_frames']
                except Exception as exc:
                    state['failed'] += 1
                    result = {'output_relative': job['output_relative'], 'status': 'FAILED', 'error': str(exc)}
                ledger.write(json.dumps(result) + '\n')
                ledger.flush()
            state['active'] = [j['output_relative'] for j in pending.values()]
            save()
    state['status'] = 'complete' if not state['failed'] else 'needs_review'
    save()
    if state['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
