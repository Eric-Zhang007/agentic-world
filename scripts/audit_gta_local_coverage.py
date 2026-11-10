"""Account for local published segments without consulting cloud inventories."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root
    source = root / 'source'
    episodes = {e['path']: e['episode_id'] for e in json.loads((source / 'data/index.json').read_bytes())['episodes']}
    jobs = json.loads((source / 'data/segments.json').read_bytes())['segments']
    rows = []
    totals = Counter()
    for job in jobs:
        episode = episodes[job['episode_relative']]
        agent, segment = Path(job['output_relative']).parts[-2:]
        key = '_'.join((episode, agent, segment))
        folder = source / 'data' / job['output_relative']
        checks = {
            'rgb_present': (source / 'data' / job['rgb_relative']).is_file(),
            'frames_present': (folder / 'frames.jsonl').is_file(),
            'manifest_present': (folder / 'manifest.json').is_file(),
            'validation_present': (folder / 'validation.json').is_file(),
            'record_index_receipt_present': (root / 'record_adapter' / (episode + '.json')).is_file(),
            'proxy_receipt_present': (root / 'proxy_cache' / job['output_relative'] / 'local_verified.json').is_file(),
            'prepared': (root / 'passages/qualification' / key / 'summary.json').is_file(),
        }
        row = {'segment_key': key, 'episode_id': episode, 'rgb_relative': job['rgb_relative'], **checks}
        row['missing_local_inputs'] = [k for k, v in checks.items() if not v and k != 'prepared']
        rows.append(row)
        totals.update({k: int(v) for k, v in checks.items()})
    out = root / 'local_coverage'
    out.mkdir(exist_ok=True)
    (out / 'segments.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    report = {'declared_segments': len(rows), 'counts': dict(totals),
              'unprepared_with_all_checked_inputs': sum(not r['prepared'] and not r['missing_local_inputs'] for r in rows),
              'missing_input_patterns': dict(Counter(','.join(r['missing_local_inputs']) for r in rows if r['missing_local_inputs'])),
              'scope': 'Local file/receipt presence only, not media integrity or training acceptance. No OSS inventory used.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
