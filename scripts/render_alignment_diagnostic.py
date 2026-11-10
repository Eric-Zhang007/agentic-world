"""Render candidate landmarks and camera revisions, checking Godot projection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from whitebox.image_alignment import landmark_points, project_points
from whitebox.camera_reconcile import update_camera


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--use-display',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for case in ('urban','rural'):
        fitted=ROOT/'outputs/local_landmark_fit_20261109'/case
        data=json.loads((fitted/'report.json').read_text())['annotations']
        paths=[]
        for name, filename in [('before','scene_before.json'),('after','scene.json')]:
            scene=json.loads((fitted/filename).read_text())
            scene['projection_check_points']=landmark_points(scene,data['landmarks']).tolist()
            paths.append((case+'_'+name,scene,data))
        if case=='urban':
            base=deepcopy(paths[-1][1])
            base['camera']['intrinsics']=dict(width=1280,height=720,fx=900,fy=800,cx=640,cy=360)
            base['camera']['up']=[.1,1,.04]
            paths.append(('calibrated_before',base,data))
            revised=deepcopy(base)
            revised['camera']=update_camera(base['camera'],{'fov':65})
            paths.append(('calibrated_after',revised,data))
        panels=[]
        with Image.open(ROOT/'outputs/gta_pi_loop_20261102'/case/'input.png') as im:
            panels.append(('Source RGB',im.convert('RGB')))
        for label,scene,annotations in paths:
            folder=a.output/label
            folder.mkdir()
            scene_path=folder/'scene.json'
            scene_path.write_text(json.dumps(scene,indent=2))
            command=[sys.executable,'-m','whitebox.godot.render','--scene',str(scene_path),
                     '--output',str(folder/'render'),'--width','1280','--height','720']
            if a.use_display:command.append('--use-display')
            start=time.perf_counter()
            with (folder/'render.log').open('w') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            report=dict(case=label,seconds=time.perf_counter()-start,returncode=result.returncode)
            if result.returncode:
                report['error']=(folder/'render.log').read_text()[-2000:]
            else:
                manifest=json.loads((folder/'render/manifest.json').read_text())
                image=next(i for i in manifest['images'] if i['file']=='camera.png')
                expected,valid=project_points(scene['projection_check_points'],scene['camera'],1280,720)
                actual=np.asarray(image['projection_check_pixels'])
                report['max_projection_error_px']=float(np.linalg.norm(actual[valid]-expected[valid],axis=1).max())
                report['source_annotation_errors_px']=np.linalg.norm(actual-np.asarray([r['pixel'] for r in annotations['landmarks']]),axis=1).tolist()
                report['renderer']=manifest['renderer']
                with Image.open(folder/'render/camera.png') as im:
                    panels.append((label,im.convert('RGB')))
            reports.append(report)
            print(json.dumps(report),flush=True)
        sheet=Image.new('RGB',(640*len(panels),390),'#222222')
        for i,(label,im) in enumerate(panels):
            sheet.paste(ImageOps.contain(im,(640,360)),(i*640,30))
            ImageDraw.Draw(sheet).text((i*640+8,8),label,fill='white')
        sheet.save(a.output/(case+'_comparison.jpg'))
    (a.output/'report.json').write_text(json.dumps(reports,indent=2))
    if any(r['returncode'] or r.get('max_projection_error_px',100)>1 for r in reports):
        raise SystemExit('Render or projection parity failed; inspect report.json')


if __name__=='__main__':main()
