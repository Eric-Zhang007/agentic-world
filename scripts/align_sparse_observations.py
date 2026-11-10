"""Generate isolated sparse maintenance candidates and image evidence."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from whitebox.observation_alignment import align_observation
from whitebox.image_alignment import landmark_points, project_points
from scripts.diagnose_mask_alignment import diagnose


def run_sequence(scene, annotations, output):
    output = Path(output)
    for observation in annotations['frames']:
        if ('perception' in observation) != ('mapping' in observation):
            raise ValueError('Perception and mapping must be supplied together')
    output.mkdir(parents=True,exist_ok=False)
    reports = []
    previous = -float('inf')
    for index, observation in enumerate(annotations['frames']):
        if observation['timestamp_sec'] <= previous:
            raise ValueError('Observation timestamps must increase')
        previous = observation['timestamp_sec']
        started = time.perf_counter()
        candidate, report = align_observation(scene,observation)
        report['seconds'] = time.perf_counter()-started
        report['image'] = observation['image']
        report['provenance'] = annotations['provenance']
        directory = output / ('step_%02d' % (index+1))
        directory.mkdir()
        (directory/'scene_before.json').write_text(json.dumps(scene,indent=2))
        (directory/'scene.json').write_text(json.dumps(candidate,indent=2))
        with Image.open(observation['image']) as source:
            image = source.convert('RGB')
        if image.size != (observation['width'],observation['height']):
            raise ValueError('Source image dimensions mismatch')
        if 'perception' in observation:
            bundle=Path(observation['perception'])
            with Image.open(bundle.parent/'input.png') as source:
                if not np.array_equal(np.asarray(source.convert('RGB')),np.asarray(image)):
                    raise ValueError('Perception bundle belongs to a different source frame')
            visual,overlay=diagnose(candidate,json.loads(bundle.read_text()),observation['mapping'],bundle.parent)
            (directory/'visual_report.json').write_text(json.dumps(visual,indent=2))
            Image.fromarray(overlay[:,:,::-1]).save(directory/'visual_overlay.jpg')
            report['visual_review']=dict(counts=visual['counts'],geometry_errors=visual['geometry_errors'],
                source_perception=str(bundle),perception_invoked=False)
        (directory/'report.json').write_text(json.dumps(report,indent=2))
        draw = ImageDraw.Draw(image)
        rows = observation['static_landmarks']
        uv,_ = project_points(landmark_points(candidate,rows),candidate['camera'],*image.size)
        for row,point in zip(rows,uv):
            x,y = row['pixel']
            draw.ellipse((x-5,y-5,x+5,y+5),outline='white',width=2)
            u,v = point
            draw.line((x,y,u,v),fill='lime',width=3)
            draw.text((x+8,y),row['id'],fill='lime',stroke_width=1,stroke_fill='black')
        skipped={row['id'] for row in report.get('skipped_actor_roots',[])}
        for row in observation.get('actor_roots',[]):
            if row.get('pixel') is None:
                continue
            x,y = row['pixel']
            color='orange' if row['id'] in skipped else 'yellow'
            label=' skipped: '+row.get('contact_visibility','unknown') if row['id'] in skipped else ' inferred ground root'
            draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
            draw.text((x+8,y),row['id']+label,fill=color,stroke_width=1,stroke_fill='black')
        image.save(directory/'overlay.jpg')
        reports.append(report)
        scene = candidate
    (output/'report.json').write_text(json.dumps(reports,indent=2))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('scene','annotations','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    reports=run_sequence(json.loads(args.scene.read_text()),json.loads(args.annotations.read_text()),args.output)
    print(json.dumps(reports,indent=2))


if __name__ == '__main__':
    main()
