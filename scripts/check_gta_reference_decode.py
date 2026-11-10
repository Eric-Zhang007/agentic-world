"""Check exported reference pixels against exact sequential source decoding."""
import json
from pathlib import Path
import cv2
import numpy as np
from memory.gta_release_media import cache_frames,cached_frame


def main():
    root=Path('E:/GTA_memory_release_20261018')
    windows={w['sample_id']:w for p in (root/'qualification').glob('*/windows.jsonl') for w in map(json.loads,p.read_text().splitlines())}
    checks=[]
    for path in sorted((root/'dataset/samples').glob('*/sample.json')):
        window=windows[path.parent.name]
        cache=cache_frames(window,root)
        actual=cv2.imread(str(path.parent/'references/rgb.png'))
        expected=cached_frame(cache,'rgb_video',window['reference_frame'])
        same=actual is not None and actual.shape==expected.shape and np.array_equal(actual,expected)
        checks.append({'sample_id':window['sample_id'],'exact_reference_pixels':bool(same)})
    (root/'reference_decode_check.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({'checked':len(checks),'mismatches':[c['sample_id'] for c in checks if not c['exact_reference_pixels']]}))


if __name__=='__main__':main()
