"""Download source images for preproduction, never insert them into world memory."""
import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    parser.add_argument('--sources', type=Path, default=Path('input_data/crossover_demo/sources.json'))
    args = parser.parse_args()
    root = args.sources.parent
    session = requests.Session()
    session.trust_env = False
    if args.proxy:
        session.proxies.update(http=args.proxy, https=args.proxy)
    session.headers['User-Agent'] = 'Mozilla/5.0'
    sources = json.loads(args.sources.read_text(encoding='utf-8'))
    candidates = list(sources['images'])
    failures = []
    for source in sources['steam']:
        try:
            response = session.get('https://store.steampowered.com/api/appdetails',
                                   params={'appids': source['appid']}, timeout=25)
            response.raise_for_status()
            data = response.json()[str(source['appid'])]['data']
            for shot in data['screenshots'][:source['count']]:
                candidates.append({'id': f"{source['region']}_{shot['id']:02d}",
                                   'region': source['region'], 'role': 'visual_candidate',
                                   'source_page': f"https://store.steampowered.com/app/{source['appid']}/",
                                   'image_url': shot['path_full'], 'publisher_source': True})
        except Exception as exc:
            failures.append({'source': source, 'error': str(exc)})
    assets = []
    for row in candidates:
        try:
            folder = root / 'references' / row['region']
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / (row['id'] + '.jpg')
            if not target.exists():
                response = session.get(row['image_url'], timeout=30)
                response.raise_for_status()
                picture = Image.open(BytesIO(response.content))
                picture.load()
                # Preserve downloaded bytes and their original image format.
                suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(picture.format)
                if not suffix:
                    raise ValueError(f'Unsupported image format: {picture.format}')
                target = folder / (row['id'] + suffix)
                if not target.exists():
                    target.write_bytes(response.content)
            with Image.open(target) as picture:
                width, height = picture.size
            assets.append(dict(row, file=target.relative_to(root).as_posix(), width=width,
                               height=height, hd_1080=width >= 1920 and height >= 1080,
                               status='UNREVIEWED_VISUAL_REFERENCE', world_fact=False,
                               h3_direct_input=False))
            print(f"{row['id']}: {width}x{height}", flush=True)
        except Exception as exc:
            failures.append({'source': row, 'error': str(exc)})
            print(f"FAILED {row['id']}: {exc}", flush=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'no_upscaling': True, 'assets': assets, 'failures': failures}
    (root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for region in sorted({row['region'] for row in assets}):
        rows = [row for row in assets if row['region'] == region]
        sheet = Image.new('RGB', (960, ((len(rows) + 2) // 3) * 212), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            x, y = i % 3 * 320, i // 3 * 212
            with Image.open(root / row['file']) as picture:
                tile = ImageOps.contain(picture.convert('RGB'), (320, 180))
                sheet.paste(tile, (x, y + 30))
            draw.text((x + 4, y + 5), f"{row['id']} {row['width']}x{row['height']}", fill='black')
        sheet.save(root / f'{region}_contact_sheet.jpg')
    print(json.dumps({'downloaded': len(assets), 'failed': len(failures)}))


if __name__ == '__main__':
    main()
