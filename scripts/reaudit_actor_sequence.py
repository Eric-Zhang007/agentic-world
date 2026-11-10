"""Generate unwarped, frame-index-labelled source evidence for temporal review."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_geometry_trial import decode_frames, save_image


def main():
    out = ROOT / 'outputs/dataset_reaudit/actor_sequence'
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / 'outputs/geometry_source_audit_20261103/audit.json').read_text())
    indices = list(range(420, 601, 12))
    timing=json.loads((out/'motion_timing.json').read_text()) if (out/'motion_timing.json').exists() else {}
    affine=timing.get('affine_timing')
    fitted={i:round(affine['scale']*i+affine['offset']) for i in indices} if affine else {}
    rgb_path = Path('E:/GTA_555_complete_rebuild_20261025/source/data/09182/agents/agent_000/video_00000.mkv')
    rgb = decode_frames(rgb_path, indices)
    proxy = decode_frames(Path(audit['proxy_video']), sorted(set(indices + [i+13 for i in indices]+list(fitted.values()))))
    pages = []
    for start in range(0, len(indices), 4):
        canvas = np.full((4 * 300, 1280, 3), 245, dtype=np.uint8)
        for row, index in enumerate(indices[start:start+4]):
            # Fixed crop retains the same image coordinates in both streams.
            for col, stream in enumerate((rgb, proxy)):
                crop = stream[index][240:520, 400:1040]
                canvas[row*300+20:(row+1)*300, col*640:(col+1)*640] = crop
                label = f'{"RGB" if col == 0 else "PROXY"} source frame {index} | crop x400:1040 y240:520'
                cv2.putText(canvas, label, (col*640+5, row*300+15), cv2.FONT_HERSHEY_SIMPLEX, .40, (0,0,0), 1)
        name = out / f'frames_{indices[start]:06d}.jpg'
        save_image(name, canvas)
        pages.append(str(name))
    if affine:
        for start in range(0,len(indices),4):
            canvas=np.full((1200,1920,3),245,dtype=np.uint8)
            for row,index in enumerate(indices[start:start+4]):
                for col,(label,frame) in enumerate((('RGB',rgb[index]),('PROXY original',proxy[index]),
                        (f'PROXY timing candidate {fitted[index]}',proxy[fitted[index]]))):
                    canvas[row*300+20:(row+1)*300,col*640:(col+1)*640]=frame[240:520,400:1040]
                    cv2.putText(canvas,f'{label} | RGB {index}',(col*640+5,row*300+15),
                                cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
            name=out/f'affine_diagnostic_{indices[start]:06d}.jpg'
            save_image(name,canvas)
            pages.append(str(name))
    for start in range(0, len(indices), 4):
        canvas = np.full((4 * 300, 1920, 3), 245, dtype=np.uint8)
        for row, index in enumerate(indices[start:start+4]):
            for col, (label, frame) in enumerate((('RGB',rgb[index]),('PROXY same index',proxy[index]),
                                                   ('PROXY +13 frames (diagnostic)',proxy[index+13]))):
                canvas[row*300+20:(row+1)*300,col*640:(col+1)*640]=frame[240:520,400:1040]
                cv2.putText(canvas,f'{label} | RGB {index}',(col*640+5,row*300+15),
                            cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
        name=out/f'lag_diagnostic_{indices[start]:06d}.jpg'
        save_image(name,canvas)
        pages.append(str(name))
    (out / 'evidence.json').write_text(json.dumps({'rgb':str(rgb_path), 'proxy':audit['proxy_video'],
        'source_indices':indices, 'pages':pages, 'interpretation':'Same decoder indices, not proof of synchronized acquisition.'}, indent=2))
    print(json.dumps({'pages':pages}))


if __name__ == '__main__':
    main()
