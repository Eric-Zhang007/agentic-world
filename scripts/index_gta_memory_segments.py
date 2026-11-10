"""Inventory available paired segments and exact camera-valid runs, read-only."""
import argparse
import json
from collections import Counter
from pathlib import Path
from memory.gta_validation import camera_exclusion


def valid_runs(rows):
    runs = []
    current = []
    rejected = Counter()
    for row in rows:
        exclusion = camera_exclusion(row)
        if exclusion:
            if current:
                runs.append(current); current = []
            rejected[exclusion] += 1
            continue
        if current:
            prev = current[-1]
            continuous = (row['proxy_frame_index'] == prev['proxy_frame_index']+1
                and row['rgb_decode_index'] == prev['rgb_decode_index']+1
                and row['server_clock_domain'] == prev['server_clock_domain']
                and row['client_clock_domain'] == prev['client_clock_domain']
                and row['server_time_estimate_ns'] > prev['server_time_estimate_ns'])
            if not continuous:
                runs.append(current); current = []
        current.append(row)
    if current:
        runs.append(current)
    return [{'start_frame': r[0]['proxy_frame_index'], 'end_frame_exclusive': r[-1]['proxy_frame_index']+1,
             'frames': len(r), 'start_ns': r[0]['server_time_estimate_ns'],
             'end_ns': r[-1]['server_time_estimate_ns'], 'clock_domain': r[0]['server_clock_domain']}
            for r in runs], dict(rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', help='Restrict full frame-index scans to this directory name')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (args.output/'segments.jsonl').open('w', encoding='utf-8') as dest:
        for path in args.root.glob('**/derived/segments/*/*/manifest.json'):
            episode = path.parents[4]
            if args.episode and episode.name != args.episode:
                continue
            record = {'manifest': str(path), 'episode_root': str(episode), 'training_ready': False}
            try:
                manifest = json.loads(path.read_text(encoding='utf-8'))
                record['episode_id'] = manifest['episode_id']
                record['agent_id'] = manifest['agent_id']
                record['pair_present'] = all((episode/manifest[k]).is_file() for k in ['rgb_video','proxy_video'])
                record['camera_file_present'] = (episode/'agents'/manifest['agent_id']/'camera.parquet').is_file()
                if not record['pair_present'] or not record['camera_file_present']:
                    record['excluded_reason'] = 'missing_local_pair_or_camera_file'
                    record['declared_frames'] = manifest.get('frame_count')
                    counts['segments'] += 1
                    counts['excluded_segments_missing_local_dependencies'] += 1
                    dest.write(json.dumps(record,ensure_ascii=False)+'\n')
                    continue
                with (path.parent/'frames.jsonl').open(encoding='utf-8') as source:
                    runs, rejected = valid_runs(json.loads(line) for line in source if line.strip())
                record.update({'camera_metadata_valid_runs':runs,'rejected_frames':rejected,
                    'remaining_checks':['actual camera/world joins','decode completeness','spatial alignment',
                                        'object visibility','state-compatible historical references']})
                counts['segments'] += 1
                counts['camera_metadata_valid_frames'] += sum(r['frames'] for r in runs)
                counts['rejected_frames'] += sum(rejected.values())
                counts['segments_with_pair_and_camera_file'] += record['pair_present'] and record['camera_file_present']
            except (OSError, ValueError, KeyError) as exc:
                record['error'] = str(exc)
                counts['errors'] += 1
            dest.write(json.dumps(record,ensure_ascii=False)+'\n')
            if counts['segments'] % 25 == 0:
                print(dict(counts),flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(counts),indent=2),encoding='utf-8')
    print(dict(counts))


if __name__ == '__main__':
    main()
