"""Validate raw-index scene sources and decode exact historical sample frames."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sqlite3
import time
import cv2
import numpy as np
import orjson
from memory.gta_release import Records,entity_map,interpolate_object
from memory.gta_validation import camera_exclusion
from memory.gta_release_media import open_video,source_media_path,rgb_file_index
from scripts.prepare_gta_object_memory import camera_geometry,project_box


def work(args):
    started=time.monotonic()
    cv2.setNumThreads(1)
    src,index_root,output=args;output=Path(output);key=src['episode_id']+'_'+src['agent_id']+'_'+src['segment']
    out=output/'qualification'/key;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return orjson.loads((out/'summary.json').read_bytes())
    print('PREPARE_RECORDS',key,flush=True)
    path=Path(src['source_manifest']);manifest=orjson.loads(path.read_bytes())
    rows=[orjson.loads(l) for l in (path.parent/'frames.jsonl').read_bytes().splitlines()]
    records=Records(Path(index_root)/src['record_index'])
    camera_keys=set(records.db.execute('SELECT domain,seq FROM camera WHERE agent=?',(src['agent_id'],)).fetchall())
    world_keys=set(records.db.execute('SELECT domain,seq FROM world').fetchall())
    errors={};points=[];cache=Path(src.get('frame_cache_root',output/'review_frames'))/key;cache.mkdir(parents=True,exist_ok=True)
    for row in rows:
        i=row['proxy_frame_index'];reason=camera_exclusion(row);data=row.get('camera_interpolation')
        if not reason:
            if not data or any((row.get('client_clock_domain'),data.get(k)) not in camera_keys for k in ('lower_client_sample_seq','upper_client_sample_seq')):reason='MISSING_CAMERA_RECORD'
            elif (row.get('server_clock_domain'),row.get('world_capture_tick')) not in world_keys:reason='MISSING_WORLD_RECORD'
        if reason:errors[str(i)]=reason;continue
        if i%120:continue
        try:
            lower,upper=[records.camera(src['agent_id'],row['client_clock_domain'],data[k]) for k in ('lower_client_sample_seq','upper_client_sample_seq')]
            if any(c['episode_id']!=src['episode_id'] or c['server_clock_domain']!=row['server_clock_domain'] or c.get('view_available') is False for c in (lower,upper)):raise ValueError('Camera identity/view mismatch')
            world=records.world(row['server_clock_domain'],row['world_capture_tick'])
            if world['episode_id']!=src['episode_id']:raise ValueError('World episode mismatch')
            origin,basis=camera_geometry(lower,upper,data['alpha'])
            if not np.isfinite(origin).all() or not np.isfinite(basis).all():raise ValueError('Nonfinite camera')
            boxes=[];maps=[entity_map(c) for c in (lower,upper)]
            for oid in sorted(maps[0].keys() & maps[1].keys()):
                try:
                    obj=interpolate_object(maps[0][oid],maps[1][oid],data['alpha'])
                    box=project_box(obj,origin,basis,*manifest['resolution'])
                    if box is not None:boxes.append(box)
                except (KeyError,TypeError,ValueError):pass
            points.append({'point_id':key+f'_f{i:06d}','episode_id':src['episode_id'],'agent_id':src['agent_id'],
                'segment':src['segment'],'time_ns':row['server_time_estimate_ns'],'server_clock_domain':row['server_clock_domain'],
                'client_clock_domain':row['client_clock_domain'],'camera_position':origin.tolist(),'forward':basis[:,2].tolist(),
                'dynamic_boxes':boxes,'resolution':manifest['resolution'],'frame_index':i,'rgb_decode_index':rgb_file_index(row),
                'source_rgb_decode_index':row['rgb_decode_index'],
                'source_manifest':str(path),'source_root':src['episode_root'],'cache_root':str(cache),
                'media_paths':src.get('media_paths',{})})
        except (KeyError,ValueError,TypeError,np.linalg.LinAlgError) as exc:errors[str(i)]=str(exc)
    records.close();counts={};media_error=None
    print('PREPARE_GEOMETRY',key,'points',len(points),'invalid_frames',len(errors),flush=True)
    by_index={r['proxy_frame_index']:r for r in rows}
    for kind in ('rgb_video','proxy_video'):
        print('PREPARE_DECODE',key,kind,flush=True)
        selected={p['rgb_decode_index'] if kind=='rgb_video' else p['frame_index']:(p['frame_index'],'png') for p in points}
        for p in points:
            end=by_index.get(p['frame_index']+119)
            if end:selected[rgb_file_index(end) if kind=='rgb_video' else end['proxy_frame_index']]=(end['proxy_frame_index'],'jpg')
        cap=open_video(source_media_path(src,manifest,kind));n=0;written=set()
        try:
            while True:
                if not cap.grab():break
                if n in selected or n==0:
                    ok,image=cap.retrieve()
                    if not ok:raise ValueError('Frame retrieval failed')
                    if list(image.shape[1::-1])!=manifest['resolution']:media_error='RESOLUTION_MISMATCH'
                if n in selected:
                    idx,ext=selected[n]
                    if not cv2.imwrite(str(cache/f'{kind}_{idx:06d}.{ext}'),image):raise OSError('Image write failed')
                    written.add(n)
                n+=1
        finally:cap.release()
        counts[kind]=n
        if set(selected)-written:media_error='MISSING_SELECTED_FRAMES'
    if counts['proxy_video']!=len(rows) or any(rgb_file_index(r)>=counts['rgb_video'] for r in rows):media_error='MEDIA_COUNT_MISMATCH'
    if media_error:points=[]
    (out/'frame_errors.json').write_bytes(orjson.dumps(errors))
    (out/'points.jsonl').write_bytes(b''.join(orjson.dumps(p)+b'\n' for p in points))
    result={'segment_key':key,'source':src,'status':'EXCLUDED_'+media_error if media_error else 'QUALIFIED',
            'decoded_frames':counts,'invalid_frames':len(errors),'indexed_points':len(points),
            'elapsed_seconds':time.monotonic()-started}
    (out/'summary.json').write_bytes(orjson.dumps(result));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--index-root',type=Path,default=Path('D:/GTA_spool_integration_20261020'))
    p.add_argument('--output',type=Path,default=Path('D:/GTA_spool_passages_20261020'));p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    rows=[orjson.loads(l) for l in (a.index_root/'source_ledger.jsonl').read_bytes().splitlines()];summaries=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(work,[(r,str(a.index_root),str(a.output)) for r in rows]),1):
            summaries.append(result)
            if i%10==0:print('PREPARED',i,'/',len(rows),'points',sum(r['indexed_points'] for r in summaries),flush=True)
    with (a.output/'trajectory_points.jsonl').open('wb') as out:
        for r in summaries:out.write((a.output/'qualification'/r['segment_key']/'points.jsonl').read_bytes())
    report={'segments':len(rows),'statuses':dict(Counter(r['status'] for r in summaries)),
            'points':sum(r['indexed_points'] for r in summaries),'invalid_frames':sum(r['invalid_frames'] for r in summaries),
            'decoded_rgb_frames':sum(r['decoded_frames']['rgb_video'] for r in summaries),
            'decoded_proxy_frames':sum(r['decoded_frames']['proxy_video'] for r in summaries)}
    (a.output/'preparation_report.json').write_bytes(orjson.dumps(report,option=orjson.OPT_INDENT_2));print(report,flush=True)


if __name__=='__main__':main()
