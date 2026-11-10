"""Audit saved real observation geometry without changing existing decisions."""
import argparse
from collections import Counter
import json
from pathlib import Path

from memory.gta_visibility import visibility_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    with (args.output/'evidence.jsonl').open('w', encoding='utf-8') as dest:
        for path in sorted(args.root.glob('*/observations.json')):
            rows = json.loads(path.read_text(encoding='utf-8'))
            counts['source_packages'] += 1
            for row in rows:
                counts['saved_observations'] += 1
                entities = row['interpolated_entities']
                for obj in entities:
                    evidence = visibility_evidence(obj, row['camera_origin'], row['ray_basis'], entities)
                    counts[evidence['geometry_status']] += 1
                    counts['vehicle_occupant_observations'] += evidence['in_vehicle']
                    record = {'source': str(path), 'frame': row['proxy_frame_index'],
                              'entity_id': str(obj['dataset_entity_id']), **evidence}
                    dest.write(json.dumps(record)+'\n')
    summary = {'counts': dict(counts), 'scope': 'Saved pilot observations, including overlapping packages; not unique footage or full dataset',
               'rgb_visibility_verified': False, 'training_ready': False}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
