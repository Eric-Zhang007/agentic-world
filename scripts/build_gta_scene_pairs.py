"""Mine actual scene crossings and export RGB/whitebox pairs without an API."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import orjson

from memory.gta_release import Records, entity_map, interpolate_object
from memory.gta_validation import camera_exclusion
from memory.gta_release_media import cached_frame, open_video
from memory.gta_scene_pairs import mine_pairs, background_evidence
from scripts.prepare_gta_object_memory import camera_geometry, project_box


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def index_points(release, output):
    destination = output/'trajectory_points.jsonl'
    counts = Counter()
    records, current = None, None
    points = [orjson.loads(line) for line in destination.read_bytes().splitlines()] if destination.exists() else []
    indexed_segments={(p['episode_id'],p['agent_id'],p['segment']) for p in points}
    try:
        for number, summary_path in enumerate(sorted((release/'qualification').glob('*/summary.json'))):
            summary = orjson.loads(summary_path.read_bytes())
            if summary['status'] != 'QUALIFIED':
                counts['excluded_segments'] += 1
                continue
            src = summary['source']
            if (src['episode_id'],src['agent_id'],src['segment']) in indexed_segments:
                continue
            if current != src['record_index']:
                if records: records.close()
                current = src['record_index']; records = Records(release/current)
            manifest_path = Path(src['source_manifest'])
            manifest = orjson.loads(manifest_path.read_bytes())
            errors_path = summary_path.parent/'frame_errors.json'
            invalid = orjson.loads(errors_path.read_bytes()) if errors_path.exists() else {}
            cache = release/'review_frames'/summary['segment_key']
            if not (cache/'complete.json').exists():
                # Former object filtering must not exclude background-only scenes.
                cache = output/'additional_frames'/summary['segment_key']
                cache.mkdir(parents=True,exist_ok=True)
                if not (cache/'complete.json').exists():
                    selected=[]
                    with (manifest_path.parent/'frames.jsonl').open('rb') as f:
                        for line in f:
                            r=orjson.loads(line)
                            if (r['proxy_frame_index']%120==0 and str(r['proxy_frame_index']) not in invalid
                                    and not camera_exclusion(r)):selected.append(r)
                    if not selected:
                        counts['no_valid_camera_samples_segments'] += 1
                        continue
                    for kind in ('rgb_video','proxy_video'):
                        mapping={r['rgb_decode_index'] if kind=='rgb_video' else r['proxy_frame_index']:r['proxy_frame_index'] for r in selected}
                        cap=open_video(Path(src['episode_root'])/manifest[kind]);frame=0;written=0
                        try:
                            while True:
                                ok,image=cap.read()
                                if not ok:break
                                if frame in mapping:
                                    if not cv2.imwrite(str(cache/f'{kind}_{mapping[frame]:06d}.png'),image):raise OSError('Cache write failed')
                                    written+=1
                                frame+=1
                        finally:cap.release()
                        if written!=len(mapping):raise ValueError('Missing source frame during sequential extraction')
                    save(cache/'complete.json',{'decode_method':'sequential exact decoder indices'})
                counts['additional_cache_segments'] += 1
            with (manifest_path.parent/'frames.jsonl').open('rb') as source:
                for line in source:
                    row = orjson.loads(line)
                    index = row['proxy_frame_index']
                    if index % 120: continue
                    counts['sampled_points'] += 1
                    if str(index) in invalid or camera_exclusion(row):
                        counts['invalid_frame'] += 1; continue
                    if not all((cache/f'{kind}_{index:06d}.png').exists() for kind in ('rgb_video','proxy_video')):
                        counts['missing_cached_frame'] += 1; continue
                    try:
                        data = row['camera_interpolation']
                        lower, upper = [records.camera(src['agent_id'], row['client_clock_domain'], data[k])
                                        for k in ('lower_client_sample_seq','upper_client_sample_seq')]
                        if any(c['episode_id']!=src['episode_id'] or c['server_clock_domain']!=row['server_clock_domain'] for c in (lower,upper)):
                            raise ValueError('Camera identity mismatch')
                        if any(c.get('view_available') is False for c in (lower, upper)):
                            raise ValueError('Unavailable camera')
                        origin, basis = camera_geometry(lower, upper, data['alpha'])
                        world = records.world(row['server_clock_domain'], row['world_capture_tick'])
                        if world['episode_id'] != src['episode_id']: raise ValueError('World identity mismatch')
                        boxes = []
                        maps = [entity_map(c) for c in (lower,upper)]
                        for oid in maps[0].keys() & maps[1].keys():
                            try:
                                obj = interpolate_object(maps[0][oid],maps[1][oid],data['alpha'])
                                box = project_box(obj,origin,basis,*manifest['resolution'])
                                if box is not None: boxes.append(box)
                            except (ValueError,KeyError,TypeError): pass
                        pid = summary['segment_key']+f'_f{index:06d}'
                        point = {'point_id':pid, 'episode_id':src['episode_id'], 'agent_id':src['agent_id'],
                                 'segment':src['segment'], 'time_ns':row['server_time_estimate_ns'],
                                 'server_clock_domain':row['server_clock_domain'], 'client_clock_domain':row['client_clock_domain'],
                                 'camera_position':origin.tolist(), 'forward':basis[:,2].tolist(),
                                 'ray_basis':basis.tolist(), 'dynamic_boxes':boxes, 'resolution':manifest['resolution'],
                                 'frame_index':index, 'rgb_decode_index':row['rgb_decode_index'],
                                 'source_manifest':str(manifest_path), 'source_root':src['episode_root'],
                                 'cache_root':str(cache), 'frame_state':row,
                                 'camera_lower':lower, 'camera_upper':upper}
                        points.append(point); counts['indexed_points'] += 1
                    except (ValueError,KeyError,TypeError,np.linalg.LinAlgError) as error:
                        counts['camera_or_world_error_'+type(error).__name__] += 1
            counts['indexed_segments'] += 1
            if number % 20 == 0: print('INDEX',number,dict(counts),flush=True)
    finally:
        if records: records.close()
    with destination.open('wb') as out:
        for p in points: out.write(orjson.dumps(p)+b'\n')
    counts['total_indexed_points']=len(points)
    counts['total_indexed_segments']=len({(p['episode_id'],p['agent_id'],p['segment']) for p in points})
    save(output/'index_report.json',dict(counts))
    return points


def images(point):
    root=Path(point['cache_root']); index=point['frame_index']
    rgb=cached_frame(root,'rgb_video',index)
    proxy=cached_frame(root,'proxy_video',index)
    mask=np.full(rgb.shape[:2],255,np.uint8)
    # Ignore HUD margins and projected dynamic actors for background evidence.
    h,w=mask.shape
    mask[:int(.06*h)]=0;mask[int(.90*h):]=0;mask[:,:int(.02*w)]=0;mask[:,int(.98*w):]=0
    for box in point['dynamic_boxes']:
        x0,y0,x1,y1=[int(v) for v in box]
        mask[max(0,y0):min(h,y1+1),max(0,x0):min(w,x1+1)]=0
    return rgb,proxy,mask


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    points=index_points(args.release,out)
    lookup={p['point_id']:p for p in points}
    candidates=mine_pairs(points,args.radius,args.min_age,args.max_angle,args.excursion,args.spacing)
    save(out/'mining_config.json',{'radius_world_units':args.radius,'minimum_age_sec':args.min_age,
        'maximum_angle_deg':args.max_angle,'minimum_excursion_world_units':args.excursion,
        'deduplication_spacing_sec':args.spacing,'trajectory_sampling_frames':120,
        'paid_api_calls':0,'source_release':str(args.release)})
    save(out/'candidates.json',candidates)
    print('CANDIDATES',len(candidates),flush=True)
    accepted=[];ledger=[];sheets=[]
    for i,pair in enumerate(candidates):
        ref,target=[lookup[pair[k]] for k in ('reference_point','target_point')]
        key=f'scene_pair_{i:05d}'
        rrgb,rproxy,rmask=images(ref);trgb,tproxy,tmask=images(target)
        evidence=background_evidence(rrgb,trgb,rmask,tmask)
        # This threshold selects measurable background overlap, not visual perfection.
        passed=(evidence['inliers']>=20 and evidence['inlier_ratio']>=.35
                and evidence['occupied_grid_cells']>=3
                and min(evidence['reference_background_fraction'],evidence['target_background_fraction'])>=.15)
        entry={'sample_id':key,**pair,'background_evidence':evidence,
               'status':'BACKGROUND_OVERLAP_VERIFIED_AUTOMATICALLY' if passed else 'INSUFFICIENT_BACKGROUND_FEATURE_EVIDENCE'}
        ledger.append(entry)
        if passed:
            root=out/'dataset'/'samples'/key;root.mkdir(parents=True,exist_ok=True)
            for name,img in [('reference_rgb',rrgb),('reference_whitebox',rproxy),('target_rgb',trgb),
                             ('target_whitebox',tproxy),('reference_background_mask',rmask),('target_background_mask',tmask)]:
                if not cv2.imwrite(str(root/(name+'.png')),img):raise OSError('Image write failed')
            for label,p in [('reference',ref),('target',target)]:
                save(root/(label+'_state.json'),{k:v for k,v in p.items() if k!='cache_root'})
            sample={**entry,'conditions':{'reference_pair':{'rgb':'reference_rgb.png','whitebox':'reference_whitebox.png'},
                    'target_whitebox':'target_whitebox.png','prompt':'prompt.txt'},
                    'supervision':{'target_rgb':'target_rgb.png'},
                    'reference_time_ns':ref['time_ns'],'target_time_ns':target['time_ns'],
                    'object_correspondence':None,'human_reviewed':False,
                    'scope':'Scene background reference pair; automatic feature QA, no object memory labels.'}
            save(root/'sample.json',sample)
            (root/'prompt.txt').write_text('Render the target whitebox view of the same location. Use the historical RGB and its paired whitebox as reference for the visible environment appearance. Follow the target camera and geometry; do not copy the historical positions or poses of people and vehicles. Unseen surfaces are not constrained by the reference.\n',encoding='utf-8')
            tiles=[]
            for label,img in [('REFERENCE RGB',rrgb),('REFERENCE WHITEBOX',rproxy),('TARGET RGB (supervision)',trgb),('TARGET WHITEBOX',tproxy)]:
                tile=cv2.resize(img,(640,360));cv2.rectangle(tile,(0,0),(640,26),(0,0,0),-1)
                cv2.putText(tile,label,(8,19),0,.5,(255,255,255),1);tiles.append(tile)
            card=np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])])
            cv2.putText(card,f'{key} age={pair["age_sec"]:.0f}s dist={pair["camera_distance"]:.1f} angle={pair["view_angle_deg"]:.1f}',(8,710),0,.55,(0,255,255),1)
            cv2.imwrite(str(root/'comparison.jpg'),card,[cv2.IMWRITE_JPEG_QUALITY,92])
            accepted.append(sample);sheets.append((key,card))
        if i%50==0:print('MATCH',i,'accepted',len(accepted),flush=True)
    with (out/'pair_ledger.jsonl').open('wb') as f:
        for row in ledger:f.write(orjson.dumps(row)+b'\n')
    dataset=out/'dataset';dataset.mkdir(exist_ok=True)
    with (dataset/'samples.jsonl').open('wb') as f:
        for row in accepted:f.write(orjson.dumps({'sample_id':row['sample_id'],'sample_root':'samples/'+row['sample_id'],
            'episode_id':row['episode_id'],'kind':row['kind'],'age_sec':row['age_sec']})+b'\n')
    gallery=out/'contact_sheets';gallery.mkdir(exist_ok=True)
    for start in range(0,len(sheets),6):
        cards=[cv2.resize(card,(640,360)) for _,card in sheets[start:start+6]]
        while len(cards)<6:cards.append(np.zeros((360,640,3),np.uint8))
        cv2.imwrite(str(gallery/f'sheet_{start//6:03d}.jpg'),np.vstack([np.hstack(cards[j:j+2]) for j in (0,2,4)]))
    report={'status':'AUTOMATIC_SCENE_PAIR_EXTRACTION_COMPLETE','indexed_points':len(points),'candidate_pairs':len(candidates),
            'exported_pairs':len(accepted),'kinds':dict(Counter(s['kind'] for s in accepted)),
            'episodes':dict(Counter(s['episode_id'] for s in accepted)),
            'age_seconds':{'min':min((s['age_sec'] for s in accepted),default=None),'max':max((s['age_sec'] for s in accepted),default=None)},
            'paid_api_calls':0,'object_correspondence_required':False,'human_reviewed_all':False}
    save(out/'report.json',report)
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,default=Path('D:/GTA_memory_release_20261018'))
    p.add_argument('--output',type=Path,default=Path('D:/GTA_scene_pairs_20261019'))
    p.add_argument('--radius',type=float,default=5.)
    p.add_argument('--min-age',type=float,default=30.)
    p.add_argument('--max-angle',type=float,default=35.)
    p.add_argument('--excursion',type=float,default=15.)
    p.add_argument('--spacing',type=float,default=30.)
    args=p.parse_args()
    if min(args.radius,args.min_age,args.excursion,args.spacing)<=0: p.error('Thresholds must be positive')
    run(args)


if __name__=='__main__':main()
