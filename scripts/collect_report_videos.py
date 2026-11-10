"""Inventory existing video artifacts; no acceptance inference from filenames."""
import argparse
import json
from pathlib import Path
import cv2


def main(a):
    rows=[]
    for path in sorted(a.root.rglob('*.mp4')):
        cap=cv2.VideoCapture(str(path))
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
        rows.append({'path':str(path.resolve()),'relative_path':path.relative_to(a.root).as_posix(),
                     'bytes':path.stat().st_size,'readable_header':cap.isOpened(),
                     'frames':frames,'fps':fps,'duration_seconds':frames/fps if fps else None,
                     'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
        cap.release()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('Inventoried',len(rows),'MP4 artifacts; not unique generations')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
