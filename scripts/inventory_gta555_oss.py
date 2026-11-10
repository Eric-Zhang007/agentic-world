"""Inventory authorized OSS source using the installed ossutil credentials."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ossutil', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    base = [a.ossutil, 'api', 'list-objects-v2', '--bucket', 'gta-data-lqr',
            '--prefix', 'GTA_555/', '--max-keys', '1000', '--output-format', 'json',
            '--region', 'cn-beijing', '--endpoint', 'https://oss-cn-beijing.aliyuncs.com',
            '--retry-times', '2']
    token = None
    count = size = pages = 0
    groups = Counter()
    with (a.output / 'oss_objects.jsonl').open('w', encoding='utf-8') as stream:
        while True:
            cmd = base + (['--continuation-token', token] if token else [])
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # ossutil appends elapsed time after its JSON document.
            data, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
            for row in data.get('Contents', []):
                stream.write(json.dumps(row) + '\n')
                count += 1
                size += int(row['Size'])
                groups[row['Key'].split('/')[1]] += 1
            stream.flush()
            pages += 1
            state = {'status': 'listing', 'pages': pages, 'objects': count,
                     'bytes': size, 'updated_at': time.time(), 'groups': dict(groups)}
            (a.output / 'OSS_STATUS.json').write_text(json.dumps(state, indent=2))
            print(f'pages={pages} objects={count} bytes={size}', flush=True)
            if str(data.get('IsTruncated')).lower() != 'true':
                break
            token = data['NextContinuationToken']
    state['status'] = 'complete'
    (a.output / 'OSS_STATUS.json').write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
