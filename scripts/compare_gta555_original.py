"""Validate an original proxy and compare all decoded frames to a rebuild."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import av
import numpy as np
import oss2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = json.loads(a.receipt.read_text())
    episodes = json.loads((a.source / 'data/episodes.json').read_text())
    e = next(e for e in episodes if e['episode_id'] == r['episode'])
    rel = e['path'].removeprefix('data/') + '/derived/segments/' + r['agent'] + '/' + r['segment']
    validation = json.loads((a.source / 'data' / rel / 'validation.json').read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    original = a.output / 'original_proxy.mp4'
    if not original.exists():
        bucket = oss2.Bucket(oss2.Auth(os.environ['OSS_ACCESS_KEY_ID'], os.environ['OSS_ACCESS_KEY_SECRET']),
                             'https://oss-cn-beijing.aliyuncs.com', 'gta-data-lqr')
        bucket.get_object_to_file('GTA_555/' + rel + '/proxy.mp4', str(original))
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != validation['published_sha256']['proxy.mp4']:
        raise ValueError('Original proxy does not match publication validation')
    errors = []
    with av.open(str(original)) as x, av.open(r['proxy']) as y:
        left, right = iter(x.decode(video=0)), iter(y.decode(video=0))
        while True:
            f, g = next(left, None), next(right, None)
            if f is None or g is None:
                if f is not None or g is not None:
                    raise ValueError('Decoded frame counts differ')
                break
            delta = f.to_ndarray(format='rgb24').astype(np.float32) - g.to_ndarray(format='rgb24')
            errors.append(float(np.mean(delta ** 2)))
    report = {'original_publication_hash_matches': True, 'decoded_frames': len(errors),
              'rebuild': r['proxy'], 'original': str(original),
              'mean_pixel_mse': float(np.mean(errors)), 'max_frame_mse': max(errors),
              'pixel_identical': all(value == 0 for value in errors),
              'scope': 'Video equivalence, not RGB geometry or visual quality approval'}
    (a.output / 'equivalence.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
