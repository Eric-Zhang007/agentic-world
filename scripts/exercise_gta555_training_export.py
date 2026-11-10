"""Export a few real matched candidates without promoting them to a release."""
import argparse
import json
import os
from pathlib import Path
from scripts.export_gta_passage_training import export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--ffmpeg', required=True)
    a = p.parse_args()
    os.environ['FFMPEG_BINARY'] = a.ffmpeg
    points = {r['point_id']: r for r in (json.loads(l) for l in (a.root / 'points.jsonl').open())}
    rows = [json.loads(l) for l in (a.root / 'background_matches.jsonl').open()]
    selected = []
    for kind in sorted({r['kind'] for r in rows}):
        selected.append(max((r for r in rows if r['kind'] == kind), key=lambda r: r['age_sec']))
    result = []
    for row in selected[:3]:
        status = export(row, points, a.root, a.release)
        result.append({'sample_id': row['target_point'], 'kind': row['kind'], 'status': status})
        print(json.dumps(result[-1]), flush=True)
    (a.root / 'export_exercise.json').write_text(json.dumps({'candidates': len(rows), 'exported': result,
        'scope': 'Integration exercise only; visual acceptance and mixture pending'}, indent=2))


if __name__ == '__main__':
    main()
