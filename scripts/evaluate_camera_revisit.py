"""Assess executed viewpoint change separately from revisit appearance."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image,ImageDraw
import yaml

from agent.action_planner import json_call
from perception.frame_sampler import sample_frames


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def validated_review(review,return_expected):
    """A PASS cannot contradict the agent's own required sub-assessments."""
    fields=['camera_execution','return_to_start','identity_consistency',
            'static_world_consistency','occlusion_and_reappearance']
    required=['camera_execution','identity_consistency','static_world_consistency']
    if return_expected:required.append('return_to_start')
    if review.get('overall') not in {'PASS','FAIL','INCONCLUSIVE'}:
        raise ValueError('Missing camera-test verdict')
    for key in fields:
        entry=review.get(key,{})
        if entry.get('status') not in {'SUPPORTED','CONTRADICTED','UNCERTAIN'} or not entry.get('evidence'):
            raise ValueError('Missing camera-test evidence: '+key)
    if review['overall']=='PASS' and any(review[key]['status']!='SUPPORTED' for key in required):
        raise ValueError('PASS contradicts required camera/identity/world assessments')
    return review


def validate_camera_excursion(scene,timeline,minimum_degrees):
    """Check a numerical task requirement against the actual compiled path."""
    if not np.isfinite(minimum_degrees) or not 0<minimum_degrees<=180:
        raise ValueError('Requested camera excursion must be in (0,180] degrees')
    cameras=[scene['camera']]+[f['camera'] for f in timeline['frames']]
    offset=np.array([np.subtract(c['position'],c['target']) for c in cameras])
    if np.any(np.linalg.norm(offset[:,[0,2]],axis=1)<1e-6):
        raise ValueError('Cannot assess yaw excursion at zero horizontal radius')
    angles=np.unwrap(np.arctan2(offset[:,0],offset[:,2]))
    excursion=float(np.max(np.abs(np.degrees(angles-angles[0]))))
    if excursion+1e-6<minimum_degrees:
        raise ValueError(f'Plan only travels {excursion:.3f} degrees; task requires {minimum_degrees}')
    return {'required_degrees':minimum_degrees,'planned_excursion_degrees':excursion,
            'scope':'Compiled planned camera, not measured video camera'}


def revisit_features(first,last):
    """2D repeatability only; dynamic pixels are not claimed as world landmarks."""
    a=cv2.imread(str(first),cv2.IMREAD_GRAYSCALE);b=cv2.imread(str(last),cv2.IMREAD_GRAYSCALE)
    if a.shape!=b.shape:raise ValueError('Frame dimensions differ')
    detector=cv2.SIFT_create(nfeatures=2000)
    ka,da=detector.detectAndCompute(a,None);kb,db=detector.detectAndCompute(b,None)
    if da is None or db is None:return {'status':'INSUFFICIENT_FEATURES'}
    matches=[x for pair in cv2.BFMatcher().knnMatch(da,db,k=2) if len(pair)==2
             for x,y in [pair] if x.distance<.75*y.distance]
    result={'status':'MEASURED_2D_ONLY','keypoints':[len(ka),len(kb)],'ratio_matches':len(matches),
            'scope':'Whole-image SIFT repeatability; not metric pose, static-only matching, or proof of world consistency.'}
    if len(matches)<8:return result
    p=np.float32([ka[m.queryIdx].pt for m in matches]);q=np.float32([kb[m.trainIdx].pt for m in matches])
    h,mask=cv2.findHomography(p,q,cv2.RANSAC,3.)
    if mask is not None and mask.sum():
        keep=mask.ravel().astype(bool)
        result.update(inliers=int(keep.sum()),inlier_ratio=float(keep.mean()),
                      median_return_displacement_px=float(np.median(np.linalg.norm(p[keep]-q[keep],axis=1))))
    return result


def evaluate(a):
    a.output.mkdir(parents=True,exist_ok=False)
    scene=json.loads((a.plan/'scene.json').read_text());timeline=json.loads((a.plan/'timeline.json').read_text())
    action=json.loads((a.plan/'action.json').read_text())
    duration=timeline['frames'][-1]['time']-(timeline['frames'][0]['time']-1/timeline['fps'])
    frames=sample_frames(a.video,a.output/'frames',duration/12)
    indices=sorted({0,len(timeline['frames'])//2,len(timeline['frames'])-1})
    proxy=[a.preview/'camera.png']+[a.preview/f'frame_{i:06d}.png' for i in indices[1:]]
    metrics=revisit_features(frames[0]['path'],frames[-1]['path'])
    save(a.output/'revisit_features.json',metrics)
    intermediate=revisit_features(frames[0]['path'],frames[len(frames)//2]['path'])
    save(a.output/'midpoint_features.json',intermediate)
    return_expected=all(np.allclose(scene['camera'][key],timeline['frames'][-1]['camera'][key],rtol=0,atol=1e-8)
                        for key in ('position','target'))
    review=json_call(yaml.safe_load(a.config.read_text())['agent'],
        'Audit a large CAMERA viewpoint-change test, not an actor-action test. Return JSON '
        '{camera_execution:{status:SUPPORTED|CONTRADICTED|UNCERTAIN,evidence:[string]},'
        'return_to_start:{status:SUPPORTED|CONTRADICTED|UNCERTAIN,evidence:[string]},'
        'identity_consistency:{status:SUPPORTED|CONTRADICTED|UNCERTAIN,evidence:[string]},'
        'static_world_consistency:{status:SUPPORTED|CONTRADICTED|UNCERTAIN,evidence:[string]},'
        'occlusion_and_reappearance:{status:SUPPORTED|CONTRADICTED|UNCERTAIN,evidence:[string]},'
        'failures:[string],uncertainties:[string],overall:PASS|FAIL|INCONCLUSIVE}. '
        'First image is the accepted visual boundary. Next images are chronological generated video '
        'samples. Last three are the PLANNED proxy start/middle/end, not evidence. '
        'Check genuine large parallax and changing visible sides of characters and buildings; '
        'zoom, actor body turns, frozen camera or local image warp do not fulfill an orbit. '
        'Check fixed masonry/plinth/stairs/paving do not morph or relocate. Distinguish legitimate '
        'disocclusion of previously unseen surfaces from changes to already visible surfaces. '
        'Check persistent identity and staff ownership, not identical pixels or exact proxy anatomy. '
        'If a return was planned, compare beginning and end layout, appearance and viewpoint. '
        'An unchanged-looking scene is not success when the required camera movement was ignored. '
        'Never infer an exact camera angle or metres from uncalibrated video. Do not make hidden '
        'frames into observations. Whole-image feature matches are auxiliary, not ground truth. '
        'This is a diagnostic; do not modify world state.',
        {'action':action,'camera_before':scene['camera'],'camera_middle':timeline['frames'][len(timeline['frames'])//2]['camera'],
         'camera_after':timeline['frames'][-1]['camera'],'video_sample_count':len(frames),
         'sample_times':[f['timestamp'] for f in frames],'return_expected':return_expected,'return_features':metrics,
         'midpoint_features':intermediate},
        [a.reference]+[f['path'] for f in frames]+proxy)
    save(a.output/'review.json',review)
    validated_review(review,return_expected)
    for group,paths in [('video', [Path(f['path']) for f in frames]),('proxy',proxy)]:
        columns=3;rows=(len(paths)+columns-1)//columns
        sheet=Image.new('RGB',(columns*400,rows*324),'white');draw=ImageDraw.Draw(sheet)
        for i,path in enumerate(paths):
            x=i%columns*400;y=i//columns*324
            with Image.open(path) as image:
                image=image.convert('RGB');image.thumbnail((400,300));sheet.paste(image,(x,y+24))
            label=f"{frames[i]['timestamp']:.2f}s" if group=='video' else ['planned start','planned middle','planned end'][i]
            draw.text((x+6,y+5),label,fill='black')
        sheet.save(a.output/f'{group}_samples.jpg')
    print(json.dumps(review,ensure_ascii=True))
    return review


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','preview','video','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/default.yaml')
    evaluate(p.parse_args())
