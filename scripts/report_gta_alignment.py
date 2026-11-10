"""Build local camera comparison and measured collision-path review artifacts."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def main():
    root=Path('outputs/gta_asset_alignment')
    source=json.loads(Path('outputs/gta_pi_loop_20261102/urban/source.json').read_text())
    panels=[('Dataset RGB',Path(source['rgb'])),('Dataset whitebox (includes actors)',Path(source['whitebox'])),
            ('Previous camera: assumed 60 degrees',root/'region_preview.png'),
            ('Recorded camera: static geometry only',root/'calibrated_camera.png')]
    canvas=Image.new('RGB',(1280,768),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,path) in enumerate(panels):
        x,y=(i%2)*640,(i//2)*384
        draw.text((x+10,y+7),label,fill='black')
        canvas.paste(Image.open(path).convert('RGB').resize((640,360)),(x,y+24))
    canvas.save(root/'camera_comparison.jpg',quality=95)
    actual=json.loads((root/'calibrated_manifest.json').read_text())
    pixels=next(r['projection_check_pixels'] for r in actual['images'] if r['file']=='camera.png')
    error=np.linalg.norm(np.asarray(pixels)-json.loads((root/'projection_expected.json').read_text()),axis=1)
    report=json.loads((root/'surface_motion_report.json').read_text())
    summary={'projection_points':len(error),'maximum_projection_error_px':float(error.max()),
             'routes':[{'requested_delta':r['delta'],'actual_delta':np.subtract(r['end'],r['start']).tolist(),
             'valid':r['checks']['valid'],'violations':r['checks']['violations'][:2]} for r in report]}
    (root/'verification.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for i,r in enumerate(report):
        points=np.asarray([r['start']]+r['root_path'])
        axes[0].plot(points[:,0],points[:,2],label=str(i))
        axes[0].scatter(*points[-1,[0,2]],s=25)
        distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points[:,[0,2]],axis=0),axis=1))]
        axes[1].plot(distance,points[:,1],label=str(i))
    axes[0].set(xlabel='World X (m)',ylabel='World Z (m)',title='Actual root paths; route 7 stopped by geometry',aspect='equal')
    axes[1].set(xlabel='Distance travelled (m)',ylabel='Ground height (m)',title='Root follows recorded collision surfaces')
    axes[1].legend(ncol=2);fig.tight_layout();fig.savefig(root/'surface_motion_paths.png',dpi=160)
    scene=json.loads((root/'surface_motion_scene.json').read_text())
    timeline=json.loads((root/'surface_motion_7.json').read_text())
    review_camera={'position':[3,8,1],'target':[-.2,-1.7,-3.59],'fov':50}
    scene['camera']=review_camera
    scene['static_geometry']={'path':'urban_region/surfaces.bin','collision_path':'urban_region/collision.bin'}
    for frame in timeline['frames']:frame['camera']=review_camera
    (root/'collision_review_scene.json').write_text(json.dumps(scene,indent=2))
    (root/'collision_review_timeline.json').write_text(json.dumps(timeline,indent=2))
    print(json.dumps(summary))


if __name__=='__main__':main()
