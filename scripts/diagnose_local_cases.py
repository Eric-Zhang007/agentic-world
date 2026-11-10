"""Offline paired-image inspection. Outputs are evaluator-only, never agent input."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def inspect(row):
    state = json.loads(Path(row['state']).read_text())
    sizes = {}
    for key in ('rgb', 'whitebox'):
        with Image.open(row[key]) as im:
            im.load()
            sizes[key] = list(im.size)
    return dict(case=row['case'], sizes=sizes,
                dimensions_match=sizes['rgb'] == sizes['whitebox'] == state.get('resolution'),
                frame_index=state.get('frame_index'), rgb_decode_index=state.get('rgb_decode_index'),
                time_ns=state.get('time_ns'), camera_position=state.get('camera_position'),
                forward=state.get('forward'), server_clock_domain=state.get('server_clock_domain'),
                client_clock_domain=state.get('client_clock_domain'),
                pairing_limit='Shared state and folder only; independent whitebox timestamp not verified')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((a.bench/'manifest.json').read_text())
    reports = []
    sheet = Image.new('RGB', (640, 200*len(rows)), 'white')
    for i, row in enumerate(rows):
        try:
            reports.append(inspect(row))
            for j, key in enumerate(('rgb', 'whitebox')):
                with Image.open(row[key]) as im:
                    sheet.paste(ImageOps.contain(im.convert('RGB'), (320, 175)), (j*320, i*200+25))
            ImageDraw.Draw(sheet).text((4, i*200+4), row['case']+' RGB | evaluation whitebox', fill='black')
        except (OSError, ValueError, KeyError) as exc:
            reports.append(dict(case=row.get('case'), error=str(exc)))
    sheet.save(a.output/'paired.jpg')
    (a.output/'report.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
