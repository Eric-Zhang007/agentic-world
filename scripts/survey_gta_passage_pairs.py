"""Quantity-first passage survey with multiple strictly historical candidates."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import orjson
import numpy as np
from scipy.spatial import cKDTree
from memory.gta_scene_pairs import angle_degrees


def load_points(path):
    fields=('point_id','episode_id','agent_id','segment','time_ns','server_clock_domain',
            'client_clock_domain','camera_position','forward','frame_index','cache_root',
            'source_manifest','source_root','rgb_decode_index','resolution','dynamic_boxes')
    result=[]
    with path.open('rb') as source:
        for line in source:
            row=orjson.loads(line)
            result.append({**{k:row[k] for k in fields},'media_paths':row.get('media_paths',{})})
    return result


def candidates(points,radius=20.,min_age=5.,angle=90.,topk=8,excluded_targets=None):
    groups=defaultdict(list)
    for p in points:groups[p['episode_id'],p['server_clock_domain']].append(p)
    output=[]
    for group in groups.values():
        group.sort(key=lambda p:(p['time_ns'],p['point_id']))
        by_id={p['point_id']:p for p in group}
        xyz=np.asarray([p['camera_position'] for p in group]);tree=cKDTree(xyz)
        tracks=defaultdict(list)
        for i,p in enumerate(group):tracks[p['agent_id'],p['client_clock_domain']].append(i)
        for j,target in enumerate(group):
            if excluded_targets and target['point_id'] in excluded_targets:continue
            options=[]
            for i in tree.query_ball_point(xyz[j],radius):
                ref=group[i];age=(target['time_ns']-ref['time_ns'])/1e9
                if age<min_age:continue
                if abs(xyz[j,2]-xyz[i,2])>4.:continue
                view_angle=angle_degrees(ref['forward'],target['forward'])
                if view_angle>angle:continue
                distance=float(np.linalg.norm(xyz[i]-xyz[j]))
                same=ref['agent_id']==target['agent_id']
                if same and ref['client_clock_domain']!=target['client_clock_domain']:continue
                kind='cross_camera_passage'
                excursion=None
                if same:
                    indices=tracks[ref['agent_id'],ref['client_clock_domain']]
                    lo=np.searchsorted(indices,i);hi=np.searchsorted(indices,j)
                    between=indices[lo:hi+1]
                    excursion=float(np.linalg.norm(xyz[between]-xyz[j],axis=1).max())
                    contiguous=all((group[b]['time_ns']-group[a]['time_ns'])/1e9<=7.5 for a,b in zip(between,between[1:]))
                    kind=('same_camera_return' if contiguous and excursion>max(distance+3.,8.)
                          else 'nearby_history' if contiguous else 'gap_unknown')
                options.append({'reference_point':ref['point_id'],'target_point':target['point_id'],
                    'episode_id':target['episode_id'],'kind':kind,'age_sec':age,
                    'camera_distance':distance,'view_angle_deg':view_angle,'max_excursion':excursion})
            # Return visits are preferred to continuous nearby history. Retain
            # several ages/views; do not drop a target after one failed match.
            options.sort(key=lambda x:(x['kind'] in ('nearby_history','gap_unknown'),x['age_sec'],
                                       x['camera_distance']+x['view_angle_deg']/10))
            chosen=[];seen=set()
            for item in options:
                ref=by_id[item['reference_point']]
                bucket=(ref['agent_id'],int(ref['time_ns']/10_000_000_000))
                if bucket in seen:continue
                seen.add(bucket);chosen.append(item)
                if len(chosen)==topk:break
            if chosen:output.append({'target_point':target['point_id'],'options':chosen})
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,default=Path('D:/GTA_scene_pairs_complete_20261019/trajectory_points.jsonl'))
    p.add_argument('--output',type=Path,default=Path('D:/GTA_passage_pairs_20261020'))
    p.add_argument('--exclude-target-index',type=Path,help='Previously searched points; retain them as historical references')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    points=load_points(args.index)
    excluded={r['point_id'] for r in map(orjson.loads,args.exclude_target_index.read_bytes().splitlines())} if args.exclude_target_index else set()
    (args.output/'points.jsonl').write_bytes(b''.join(orjson.dumps(v)+b'\n' for v in points))
    report=[]
    for radius,age,angle in [(5,30,35),(10,10,60),(20,5,90),(30,2,110)]:
        rows=candidates(points,radius,age,angle,excluded_targets=excluded)
        entry={'radius':radius,'minimum_age':age,'max_angle':angle,'targets':len(rows),
               'candidate_edges':sum(len(r['options']) for r in rows),
               'first_choice_types':dict(Counter(r['options'][0]['kind'] for r in rows))}
        report.append(entry);print(entry,flush=True)
        name=f'candidates_r{radius}_t{age}_a{angle}.jsonl'
        (args.output/name).write_bytes(b''.join(orjson.dumps(v)+b'\n' for v in rows))
    (args.output/'survey.json').write_bytes(orjson.dumps(report,option=orjson.OPT_INDENT_2))


if __name__=='__main__':main()
