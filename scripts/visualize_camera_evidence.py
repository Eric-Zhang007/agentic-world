"""Compare candidate projections against explicit fit and held-out image evidence."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whitebox.image_alignment import landmark_points, project_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--landmarks', type=Path, required=True)
    parser.add_argument('--scene', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError('Source image could not be decoded')
    data = json.loads(args.landmarks.read_text())
    if image.shape[:2] != (data['height'], data['width']):
        raise ValueError('Source image and landmark coordinate dimensions differ')
    args.output.mkdir(parents=True, exist_ok=False)
    panels, reports = [], []
    for index, path in enumerate(args.scene):
        scene = json.loads(path.read_text())
        panel = image.copy()
        report = {'scene': str(path), 'groups': {}}
        for group, color in [('landmarks', (40, 220, 80)), ('heldout', (30, 140, 255))]:
            rows = data.get(group, [])
            if not rows:
                continue
            uv, valid = project_points(landmark_points(scene, rows), scene['camera'],
                                       data['width'], data['height'])
            entries = []
            for number, (row, pixel, ok) in enumerate(zip(rows, uv, valid)):
                observed = np.asarray(row['pixel'], dtype=float)
                error = float(np.linalg.norm(pixel-observed)) if ok else None
                entries.append({'id': row['id'], 'observed': observed.tolist(),
                                'projected': pixel.tolist() if ok else None,
                                'error_px': error})
                point = tuple(np.rint(observed).astype(int))
                cv2.circle(panel, point, 6, color, 2)
                if ok and np.isfinite(pixel).all() and np.max(np.abs(pixel)) < 1e7:
                    predicted = tuple(np.rint(pixel).astype(int))
                    cv2.drawMarker(panel, predicted, color, cv2.MARKER_CROSS, 14, 2)
                    cv2.line(panel, point, predicted, color, 2)
                label = f'{group[0]}{number} {row["id"]}: {error:.1f}px' if ok else f'{row["id"]}: behind'
                cv2.putText(panel, label, (max(0, point[0]-100), max(60, point[1]-12)),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
            report['groups'][group] = entries
        cv2.rectangle(panel, (0, 0), (image.shape[1], 43), (20, 20, 20), -1)
        cv2.putText(panel, f'{index}: {path.parent.name} | circle=evidence cross=projection',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(args.output/f'{index:02d}.jpg'), panel):
            raise OSError('Unable to save evidence image')
        panels.append(panel)
        reports.append(report)
    cv2.imwrite(str(args.output/'comparison.jpg'), np.concatenate(panels, axis=0))
    (args.output/'report.json').write_text(json.dumps({'world_commit': False,
        'source_image': str(args.image), 'candidates': reports}, indent=2))
    print(args.output/'comparison.jpg')


if __name__ == '__main__':
    main()
