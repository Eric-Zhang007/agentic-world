"""Summarize actual per-sample decisions without promoting pending candidates."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    samples = [json.loads(p.read_bytes()) for p in sorted((args.root / 'training/samples').glob('*/sample.json'))]
    accepted = [s for s in samples if s['training_status'] == 'AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA']
    report = {'scope': 'Historical paired reference and five target frames; not exhaustive temporal inspection',
              'sample_count': len(samples), 'statuses': dict(Counter(s['training_status'] for s in samples)),
              'accepted_kinds': dict(Counter(s['reference']['kind'] for s in accepted)),
              'accepted_episodes': dict(Counter(s['episode_id'] for s in accepted)),
              'accepted_reference_age_at_least_120s': sum(s['reference']['age_sec'] >= 120 for s in accepted),
              'formal_delivery': False, 'training_run': False}
    (args.root / 'visual_review_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    cards = []
    for s in samples:
        key = html.escape(s['sample_id'])
        status = html.escape(s['training_status'])
        reason = html.escape(s.get('supplemental_target_review', {}).get('reason', s.get('reason', 'Camera exclusion')))
        prefix = f'training/samples/{key}/'
        images = ''
        if (args.root / 'training/samples' / s['sample_id'] / 'video_review.jpg').exists():
            images = (f'<img loading="lazy" src="{prefix}comparison.jpg">'
                      f'<img loading="lazy" src="{prefix}video_review.jpg">'
                      f'<p><a href="{prefix}target_rgb.mp4">RGB video</a> | '
                      f'<a href="{prefix}target_whitebox.mp4">Whitebox video</a></p>')
        cards.append(f'<details><summary>{key} | {status}</summary><p>{reason}</p>{images}</details>')
    page = ('<!doctype html><meta charset="utf-8"><title>GTA paired-media review</title>'
            '<style>body{font:15px system-ui;margin:24px;background:#fff;color:#222}'
            'img{display:block;max-width:100%;margin:12px 0}details{border-bottom:1px solid #ccc;padding:12px 0}'
            'summary{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>'
            '<h1>Paired-media visual review</h1><pre>' + html.escape(json.dumps(report, indent=2)) + '</pre>'
            + ''.join(cards))
    (args.root / 'visual_review.html').write_text(page, encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
