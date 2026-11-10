"""Recheck exclusions caused by using episode-global indices on segment files."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.prepare_gta_spool_passages import work


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--indices', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--resume-watch', action='store_true', help='Resume full qualification after the repair pass')
    a = p.parse_args()
    processed = 0
    for path in sorted((a.root / 'qualification').glob('*/summary.json')):
        summary = json.loads(path.read_bytes())
        if summary['status'] != 'EXCLUDED_MEDIA_COUNT_MISMATCH':
            continue
        source = summary['source']
        frames_path = Path(source['source_manifest']).with_name('frames.jsonl')
        with frames_path.open() as stream:
            first = json.loads(next(stream))
        if 'rgb_segment_frame_index' not in first or first['rgb_segment_frame_index'] == first['rgb_decode_index']:
            continue
        archive = path.parent / 'before_local_index_fix'
        archive.mkdir(exist_ok=True)
        if (archive / 'summary.json').exists():
            raise ValueError('Prior repair already archived; inspect before retrying: ' + str(path))
        for name in ('summary.json', 'points.jsonl', 'frame_errors.json'):
            if (path.parent / name).exists():
                shutil.move(str(path.parent / name), str(archive / name))
        result = work((source, str(a.indices), str(a.root)))
        print('RECHECKED', result['segment_key'], result['status'], result['indexed_points'], flush=True)
        processed += 1
        if a.limit and processed >= a.limit:
            break
    print('RECHECK_COMPLETE', processed, flush=True)
    if a.resume_watch:
        base = a.root.parent
        subprocess.run([sys.executable, '-u', '-m', 'scripts.qualify_gta555_sources',
                        '--source', str(base / 'source'), '--old-source', 'E:/GTA_555',
                        '--indices', str(a.indices), '--proxies', str(base / 'proxy_cache'),
                        '--output', str(a.root), '--cache-root', 'D:/GTA_source_work_cache', '--watch'],
                       check=True)


if __name__ == '__main__':
    main()
