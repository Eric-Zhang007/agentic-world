"""Combine explicit detector passes on identical pixels, preserving provenance."""
import argparse
import json
from pathlib import Path
from PIL import Image


def merge(paths, output):
    records = [json.loads(p.read_text()) for p in paths]
    images = [Image.open(p.parent/'input.png').convert('RGB') for p in paths]
    if any(im.size != images[0].size or im.tobytes() != images[0].tobytes() for im in images[1:]):
        raise ValueError('Perception passes must use identical input RGB')
    combined = dict(records[0], observations=[], merged_passes=[str(p.resolve()) for p in paths])
    for index, (path, record) in enumerate(zip(paths, records)):
        for item in record['observations']:
            item = dict(item, observation_id=f'pass{index}:{item["observation_id"]}',
                        perception_pass=str(path.resolve()))
            for key in ('mask_path', 'embedding_path'):
                asset = Path(item[key])
                if not asset.is_absolute():
                    asset = path.parent/asset
                item[key] = str(asset.resolve(strict=True))
            combined['observations'].append(item)
    output.mkdir(parents=True, exist_ok=False)
    images[0].save(output/'input.png')
    (output/'result.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
    return output/'result.json'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pass', dest='passes', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(merge(a.passes, a.output))
