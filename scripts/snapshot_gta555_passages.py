"""Snapshot completed preparation records for an isolated pipeline exercise."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    counts = {'segments': 0, 'points': 0, 'scope': 'Completed subset; not full dataset'}
    with (a.output / 'trajectory_points.jsonl').open('wb') as target:
        for path in sorted(a.prepared.glob('qualification/*/summary.json')):
            record = json.loads(path.read_text())
            if record['status'] != 'QUALIFIED':
                continue
            data = (path.parent / 'points.jsonl').read_bytes()
            target.write(data)
            counts['segments'] += 1
            counts['points'] += len(data.splitlines())
    (a.output / 'snapshot.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
