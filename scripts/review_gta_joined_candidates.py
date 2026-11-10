"""Export sparse RGB/proxy review sheets from record-qualified candidates."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    with args.audit.open(encoding='utf-8') as source:
        candidates = [r for r in map(json.loads, source)
                      if r['record_join_passed'] and r['camera_supported_entity_ids']]
    if not candidates:
        raise ValueError('No record-qualified candidates')
    args.output.mkdir(parents=True, exist_ok=False)
    chosen = np.linspace(0, len(candidates)-1, min(args.count, len(candidates)), dtype=int)
    results = []
    for number, index in enumerate(chosen):
        record = candidates[int(index)]
        path = Path(record['source_manifest'])
        manifest = json.loads(path.read_text(encoding='utf-8'))
        with (path.parent/'frames.jsonl').open(encoding='utf-8') as source:
            frames = {r['proxy_frame_index']: r for r in map(json.loads, source)}
        start, end = record['target_start_frame'], record['target_end_frame_exclusive']
        selected = [record['reference_frame'], start, (start+end)//2, end-1]
        captures = [cv2.VideoCapture(str(args.episode/manifest[key])) for key in ('rgb_video', 'proxy_video')]
        cards = []
        try:
            for frame_index in selected:
                frame = frames[frame_index]
                images = []
                for cap, key in zip(captures, ('rgb_decode_index', 'proxy_frame_index')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame[key])
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f'Decode failed: {path} {key}={frame[key]}')
                    images.append(cv2.resize(image, (640, 360)))
                card = np.concatenate(images, axis=1)
                label = f'{number}: {record["agent_id"]} {record["segment"]} frame={frame_index} ' + ('REFERENCE' if frame_index == selected[0] else 'TARGET')
                cv2.rectangle(card, (0, 0), (1280, 28), (0, 0, 0), -1)
                cv2.putText(card, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                cards.append(card)
        finally:
            for cap in captures:
                cap.release()
        name = f'{number:02d}.jpg'
        if not cv2.imwrite(str(args.output/name), np.concatenate(cards, axis=0)):
            raise OSError('Failed to write review sheet')
        results.append({'sheet': name, 'sampled_frames': selected, 'candidate': record,
                        'review_status': 'PENDING', 'sparse_review_only': True})
        print(name, flush=True)
    (args.output/'review_index.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
