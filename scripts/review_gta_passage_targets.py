"""Prepare contact sheets and apply explicit sampled target-video review."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import orjson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('D:/GTA_passage_pairs_20261020'))
    parser.add_argument('--batch', required=True)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    batch = args.root / args.batch
    samples = args.root / 'training' / 'samples'
    if args.decisions is None:
        if batch.exists():
            raise ValueError('Do not overwrite a review index')
        batch.mkdir()
        keys = []
        for path in sorted(samples.glob('*/sample.json')):
            sample = orjson.loads(path.read_bytes())
            if sample['training_status'] == 'VIDEO_QA_PENDING':
                keys.append(sample['sample_id'])
                if args.limit > 0 and len(keys) >= args.limit:
                    break
        (batch / 'index.json').write_bytes(orjson.dumps(keys))
        (batch / 'layout.json').write_bytes(orjson.dumps({'historical_reference_pair': True,
            'target_frame_indices': [0,30,60,90,119]}))
        for start in range(0, len(keys), 4):
            cards = []
            for i in range(start, min(start + 4, len(keys))):
                image = cv2.imread(str(samples / keys[i] / 'video_review.jpg'))
                if image is None:
                    raise ValueError('Missing review image')
                image = cv2.resize(image, (1280, 288))
                refs = [cv2.imread(str(samples / keys[i] / name)) for name in ('reference_rgb.png','reference_whitebox.png')]
                if any(ref is None for ref in refs):
                    raise ValueError('Missing historical reference image')
                reference = np.vstack([cv2.resize(ref,(256,144)) for ref in refs])
                image = np.hstack([reference,image])
                image[:22] = 0
                cv2.putText(image, f'{i}: {keys[i]}', (5, 16), 0, .46, (255, 255, 255), 1)
                cards.append(image)
            cv2.imwrite(str(batch / f'sheet_{start // 4:03d}.jpg'), np.vstack(cards))
        print({'pending_targets': len(keys), 'batch': str(batch)})
        return
    keys = orjson.loads((batch / 'index.json').read_bytes())
    decisions = orjson.loads(args.decisions.read_bytes())
    reviewed = decisions.get('reviewed_indices', list(range(len(keys))))
    if len(reviewed) != len(set(reviewed)) or any(type(i) is not int or not 0 <= i < len(keys) for i in reviewed):
        raise ValueError('Invalid or repeated reviewed index')
    if decisions['reviewed_count'] != len(reviewed):
        raise ValueError('Review index differs from reviewed batch')
    if not decisions['all_sheets_inspected']:
        raise ValueError('Explicit inspection required')
    rejected = decisions['rejected']
    layout_path = batch / 'layout.json'
    reference_reviewed = layout_path.exists() and orjson.loads(layout_path.read_bytes()).get('historical_reference_pair',False)
    if any(int(i) not in reviewed for i in rejected):
        raise ValueError('Invalid reviewed index')
    ledger = []
    for i in reviewed:
        key = keys[i]
        path = samples / key / 'sample.json'
        sample = orjson.loads(path.read_bytes())
        scope = ('Historical RGB/whitebox reference pair and target frames 0,30,60,90,119; not exhaustive temporal review'
                 if reference_reviewed else 'Target RGB/whitebox frames 0,30,60,90,119; not exhaustive frame review or new-reference review')
        review = {'scope': scope, 'historical_reference_pair_reviewed': reference_reviewed,
                  'method': 'Codex visual inspection of contact sheets', 'batch': args.batch,
                  'accepted': str(i) not in rejected,
                  'reason': rejected.get(str(i), 'Sampled target RGB and coarse whitebox scene correspondence acceptable')}
        sample['supplemental_target_review'] = review
        if reference_reviewed:
            sample['visual_reference_review'] = 'Historical paired reference inspected alongside five sampled target frames in '+args.batch
        sample['training_status'] = ('AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA' if review['accepted']
                                     else 'EXCLUDED_SUPPLEMENTAL_TARGET_QA')
        path.write_bytes(orjson.dumps(sample, option=orjson.OPT_INDENT_2))
        ledger.append({'sample_id': key, **review})
    ledger_path = batch / 'decisions.jsonl'
    previous = {r['sample_id']: r for r in map(orjson.loads, ledger_path.read_bytes().splitlines())} if ledger_path.exists() else {}
    previous.update({r['sample_id']: r for r in ledger})
    ledger_path.write_bytes(b''.join(orjson.dumps(row) + b'\n' for row in previous.values()))
    print({'reviewed': len(reviewed), 'accepted': len(reviewed) - len(rejected), 'excluded': len(rejected),
           'remaining': len(keys) - len(previous)})


if __name__ == '__main__':
    main()
