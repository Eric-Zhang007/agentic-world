"""Merge a real road-only probe pass into an existing geometry perception run."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    original = out / 'result_joint_prompt.json'
    if not original.exists():
        shutil.copy2(out / 'result.json', original)
        shutil.copy2(out / 'embeddings.npy', out / 'embeddings_joint_prompt.npy')
        shutil.copy2(out / 'overlay.png', out / 'overlay_joint_prompt.png')
    report = json.loads(original.read_text())
    extra = json.loads((out / 'road_pass/result.json').read_text())
    assert report['image_size'] == extra['image_size'] == [1280, 720]
    assert np.array_equal(np.asarray(Image.open(out / 'input.png')),
                          np.asarray(Image.open(out / 'road_pass/input.png')))
    roads = [o for o in extra['observations'] if o['category'] == 'road']
    assert len(roads) == 1, 'Review road proposals before merging'
    indices = [i for i, o in enumerate(report['observations'])
               if o['category'] == 'road sidewalk']
    assert len(indices) == 1, 'Expected one ambiguous road candidate'
    i = indices[0]
    road = dict(roads[0])
    old = report['observations'][i]
    backup_mask = out / 'mask_joint_prompt_road.png'
    if not backup_mask.exists():
        shutil.copy2(old['mask_path'], backup_mask)
    shutil.copy2(road['mask_path'], out / f'mask_{i:03d}.png')
    features = np.load(out / 'embeddings_joint_prompt.npy')
    features[i] = np.load(out / 'road_pass/embeddings.npy')[road['embedding_index']]
    np.save(out / 'embeddings.npy', features)
    road.update(observation_id=old['observation_id'],
                mask_path=str(out / f'mask_{i:03d}.png'),
                embedding_path=str(out / 'embeddings.npy'), embedding_index=i)
    report['observations'][i] = road
    report['supplemental_pass'] = {'result': str(out / 'road_pass/result.json'),
                                  'prompt': 'road.', 'replaced_index': i,
                                  'original_result': str(original)}
    report['timings'] = {k: v + extra['timings'][k] for k, v in report['timings'].items()}
    report['peak_gpu_mib'] = max(report['peak_gpu_mib'], extra['peak_gpu_mib'])
    overlay = cv2.imread(str(out / 'input.png'))
    for j, obs in enumerate(report['observations']):
        mask = np.asarray(Image.open(obs['mask_path'])) > 0
        assert mask.shape == (720, 1280) and int(mask.sum()) == obs['mask_area']
        color = ((67 + j * 53) % 255, (173 + j * 31) % 255, (101 + j * 97) % 255)
        overlay[mask] = (overlay[mask] * .6 + np.array(color) * .4).astype(np.uint8)
        x1, y1, x2, y2 = map(int, obs['bbox_xyxy'])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{j}: {obs['category']} {obs['confidence']:.2f}",
                    (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.imwrite(str(out / 'overlay.png'), overlay)
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'observations': len(report['observations']), 'road': road}, indent=2))


if __name__ == '__main__':
    main()
