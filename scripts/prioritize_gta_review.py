"""Refresh eligibility and round-robin review sheets across episodes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def labels(row):
    result = [row['passage_kind']]
    result += ['input:' + x for x in row['sampled_active_inputs']]
    result += ['mode:' + x for x in row['recorded_modes']]
    result += ['change:' + k for k, v in row['sampled_state_changes'].items() if v]
    age = row['reference_age_sec']
    result.append('age:' + ('120+' if age >= 120 else '30-120' if age >= 30 else '10-30' if age >= 10 else '0-10'))
    angle = row.get('reference_angle_deg')
    if angle is not None:
        result.append('angle:' + ('90+' if angle >= 90 else '45-90' if angle >= 45 else '0-45'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--batch', default='pending_audit_full_source')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    batch = a.training_root / a.batch
    indices = {key: i for i, key in enumerate(json.loads((batch / 'index.json').read_bytes()))}
    rows, statuses, frequencies = [], Counter(), Counter()
    for line in a.annotations.read_text().splitlines():
        row = json.loads(line)
        sample = json.loads((a.training_root / 'training/samples' / row['sample_id'] / 'sample.json').read_bytes())
        row['current_training_status'] = sample['training_status']
        row['sampling_labels'] = labels(row)
        statuses[row['current_training_status']] += 1
        frequencies.update(row['sampling_labels'])
        rows.append(row)
    sheets = {}
    for row in rows:
        if row['current_training_status'] != 'VIDEO_QA_PENDING':
            continue
        index = indices[row['sample_id']]
        number = index // 4
        sheet = sheets.setdefault(number, {'sheet': str(batch / f'sheet_{number:03d}.jpg'),
            'episode_id': row['episode_id'], 'pending_indices': [], 'priority': 0.0})
        sheet['pending_indices'].append(index)
        sheet['priority'] = max(sheet['priority'], sum(1 / frequencies[x] for x in row['sampling_labels']))
    episodes = defaultdict(list)
    for sheet in sorted(sheets.values(), key=lambda x: (-x['priority'], x['sheet'])):
        episodes[sheet['episode_id']].append(sheet)
    queue = []
    while any(episodes.values()):
        for episode in sorted(episodes):
            if episodes[episode]:
                queue.append(episodes[episode].pop(0))
    accepted = [r for r in rows if r['current_training_status'] == 'AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA']
    for name, records in [('review_queue.jsonl', queue), ('reviewed_sampling_candidates.jsonl', accepted)]:
        (a.output / name).write_text(''.join(json.dumps(r) + '\n' for r in records))
    report = {'annotated_samples': len(rows), 'current_statuses': dict(statuses),
        'pending_sheets': len(queue), 'reviewed_candidates': len(accepted),
        'label_counts': dict(frequencies),
        'scope': 'Sampling/audit labels only. Review candidates are not a finalized split or balanced training manifest.'}
    (a.output / 'review_coverage.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
