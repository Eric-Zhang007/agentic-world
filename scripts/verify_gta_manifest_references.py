"""Check every included reference dependency without model or network calls."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixture', type=Path, required=True)
    parser.add_argument('--auxiliary-roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = {k: Path(v) for k, v in json.loads((args.mixture / 'roots.example.json').read_bytes()).items()}
    counts = Counter()
    for split in ('train', 'validation', 'test'):
        for line in (args.mixture / (split + '.jsonl')).read_text().splitlines():
            row = json.loads(line)
            folder = roots[row['dataset']] / row['sample_root']
            sample = json.loads((folder / 'sample.json').read_bytes())
            conditions = sample['conditions']
            pairs = conditions.get('reference_pairs') or [conditions['reference_pair']]
            ids = set()
            for pair in pairs:
                ref_id = pair.get('reference_id', row['sample_id'] + ':scene_ref')
                if ref_id in ids:
                    raise ValueError('Duplicate reference ID: ' + row['sample_id'])
                ids.add(ref_id)
                for name in ('rgb', 'whitebox'):
                    path = folder / pair[name]
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError('Missing or empty reference: ' + str(path))
                    counts['reference_images'] += 1
            for edge in sample.get('object_references', []):
                if edge['reference_id'] not in ids:
                    raise ValueError('Unknown reference edge: ' + row['sample_id'])
                mask = folder / edge['reference_mask']
                if not mask.is_file() or mask.stat().st_size == 0:
                    raise ValueError('Missing reference mask: ' + str(mask))
                if len(edge['target_projected_bboxes']) != sample['frame_count']:
                    raise ValueError('Projection length differs: ' + row['sample_id'])
                counts['object_reference_edges'] += 1
            if row['task'] == 'scene_background':
                auxiliary = [root / (row['sample_id'] + '.json') for root in args.auxiliary_roots
                             if (root / (row['sample_id'] + '.json')).is_file()]
                if len(auxiliary) != 1:
                    raise ValueError('Expected one scene annotation: ' + row['sample_id'])
                payload = json.loads(auxiliary[0].read_bytes())
                if payload['sample_id'] != row['sample_id'] or payload['episode_id'] != row['episode_id']:
                    raise ValueError('Auxiliary identity mismatch: ' + row['sample_id'])
                if len(payload['target_samples']) != 5:
                    raise ValueError('Auxiliary sample count differs: ' + row['sample_id'])
                counts['scene_auxiliary_annotations'] += 1
            counts['records'] += 1
    report = {'passed': True, 'counts': dict(counts),
              'scope': 'All included reference file presence, object edge links and scene auxiliary identities; no new visual approval',
              'external_review_calls': 0}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
