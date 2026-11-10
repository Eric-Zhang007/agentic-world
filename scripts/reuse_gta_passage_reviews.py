"""Reuse sampled visual decisions only for byte-identical paired media."""
import argparse
from collections import Counter
import filecmp
import json
from pathlib import Path


MEDIA = ('reference_rgb.png', 'reference_whitebox.png', 'target_rgb.png',
         'target_whitebox.png', 'target_rgb.mp4', 'target_whitebox.mp4')


def reuse(source, target):
    source_sample = json.loads((source / 'sample.json').read_bytes())
    target_path = target / 'sample.json'
    sample = json.loads(target_path.read_bytes())
    review = source_sample.get('supplemental_target_review', {})
    if sample.get('training_status') != 'VIDEO_QA_PENDING':
        return 'preserved_existing_status'
    if not review.get('historical_reference_pair_reviewed') or type(review.get('accepted')) is not bool:
        return 'no_paired_review'
    identity = ('reference_point', 'target_point')
    if any(source_sample.get('reference', {}).get(k) != sample.get('reference', {}).get(k)
           or sample.get('reference', {}).get(k) is None for k in identity):
        return 'different_pair'
    for key in ('reference_time_ns', 'target_time_ns', 'frame_count', 'fps'):
        if sample.get(key) is None or source_sample.get(key) != sample[key]:
            return 'different_timeline'
    filecmp.clear_cache()
    for name in MEDIA:
        if not (source / name).is_file() or not (target / name).is_file():
            return 'missing_media'
        if not filecmp.cmp(source / name, target / name, shallow=False):
            return 'different_media'
    sample['supplemental_target_review'] = {**review, 'reused_from': str(source / 'sample.json'),
                                            'reuse_basis': 'Exact paired media bytes and timeline'}
    sample['visual_reference_review'] = source_sample.get('visual_reference_review')
    sample['training_status'] = ('AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA' if review['accepted']
                                 else 'EXCLUDED_SUPPLEMENTAL_TARGET_QA')
    target_path.write_text(json.dumps(sample, indent=2), encoding='utf-8')
    return 'accepted' if review['accepted'] else 'excluded'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    a = p.parse_args()
    counts = Counter()
    for path in sorted((a.target / 'training/samples').glob('*/sample.json')):
        source = a.source / 'training/samples' / path.parent.name
        counts[reuse(source, path.parent) if (source / 'sample.json').exists() else 'no_previous_sample'] += 1
    report = {'counts': dict(counts), 'source': str(a.source), 'scope': 'Identical paired media only'}
    (a.target / 'review_reuse_report.json').write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == '__main__':
    main()
