"""Release eligibility and historical object references, using actual records."""
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sqlite3
import orjson
import pyarrow.parquet as pq

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from memory.gta_validation import camera_exclusion
from memory.gta_visibility import geometry_evidence
from scripts.build_gta_release_index import unpack
from scripts.prepare_gta_object_memory import camera_geometry, project_box


def model(value):
    return None if value is None else int(value) & 0xffffffff


class Records:
    def __init__(self, path):
        self.db = sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True)
        self.row_group=lru_cache(maxsize=3)(self.row_group)
        self.camera=lru_cache(maxsize=32)(self.camera)
        self.world=lru_cache(maxsize=32)(self.world)
        self.entity=lru_cache(maxsize=2048)(self.entity)

    def row_group(self,path,group):
        return pq.ParquetFile(path).read_row_group(group,columns=['payload_json']).column(0).to_pylist()

    def payload(self,value):
        result=unpack(value)
        if '_jsonl' in result:
            with open(result['_jsonl'], 'rb') as source:
                source.seek(result['offset'])
                data = source.read(result['length'])
            if len(data) != result['length']:
                raise ValueError('Truncated indexed JSONL source')
            return orjson.loads(data)
        if '_parquet' in result:
            return orjson.loads(self.row_group(result['_parquet'],result['group'])[result['row']])
        return result

    def camera(self, agent, domain, seq):
        rows = self.db.execute('SELECT data FROM camera WHERE agent=? AND domain=? AND seq=?', (agent,domain,seq)).fetchall()
        if len(rows) != 1:
            raise ValueError('MISSING_OR_AMBIGUOUS_CAMERA')
        value = self.payload(rows[0][0])
        if isinstance(value.get('camera'), dict):
            # Raw client spool nests camera fields; Parquet camera payloads flatten them.
            value = {**value, **value['camera']}
        return value

    def world(self, domain, seq):
        rows = self.db.execute('SELECT data FROM world WHERE domain=? AND seq=?', (domain,seq)).fetchall()
        if len(rows) != 1:
            raise ValueError('MISSING_OR_AMBIGUOUS_WORLD')
        return self.payload(rows[0][0])

    def entity(self, oid):
        rows = self.db.execute('SELECT data FROM entity WHERE id=?', (str(oid),)).fetchall()
        if len(rows) != 1:
            raise ValueError('MISSING_OR_AMBIGUOUS_LIFECYCLE')
        return unpack(rows[0][0])

    def close(self):
        self.camera.cache_clear(); self.world.cache_clear(); self.entity.cache_clear()
        self.row_group.cache_clear()
        self.db.close()


def entity_map(camera):
    records = (camera.get('scene_observation') or {}).get('entities', [])
    counts = Counter(str(e.get('dataset_entity_id')) for e in records)
    return {str(e['dataset_entity_id']):e for e in records
            if e.get('dataset_entity_id') is not None and counts[str(e['dataset_entity_id'])] == 1}


def interpolate_object(a, b, alpha):
    if model(a.get('model_hash')) != model(b.get('model_hash')):
        raise ValueError('MODEL_CHANGED')
    obj = dict(a)
    obj['position'] = ((1-alpha)*np.asarray(a['position'])+alpha*np.asarray(b['position'])).tolist()
    obj['orientation_quaternion'] = Slerp([0,1], Rotation.from_quat([a['orientation_quaternion'],b['orientation_quaternion']]))([alpha]).as_quat()[0].tolist()
    if a.get('current_vehicle_network_id') != b.get('current_vehicle_network_id'):
        raise ValueError('OCCUPANCY_TRANSITION')
    return obj


def frame_evidence(records, manifest, row):
    exclusion = camera_exclusion(row)
    if exclusion:
        raise ValueError(exclusion)
    interpolation = row['camera_interpolation']
    cameras = [records.camera(manifest['agent_id'],row['client_clock_domain'],interpolation[key])
               for key in ('lower_client_sample_seq','upper_client_sample_seq')]
    if any(c['episode_id'] != manifest['episode_id'] or c['server_clock_domain'] != row['server_clock_domain'] for c in cameras):
        raise ValueError('CAMERA_CLOCK_OR_EPISODE_MISMATCH')
    world = records.world(row['server_clock_domain'],row['world_capture_tick'])
    if world['episode_id'] != manifest['episode_id']:
        raise ValueError('WORLD_EPISODE_MISMATCH')
    origin, basis = camera_geometry(*cameras, interpolation['alpha'])
    if not np.isfinite(origin).all():
        raise ValueError('INVALID_CAMERA_ORIGIN')
    maps = [entity_map(c) for c in cameras]
    candidates = {}
    near_camera = []
    inverse = np.linalg.inv(basis)
    inverse_norm = np.linalg.norm(inverse,axis=1)
    for oid in maps[0].keys() & maps[1].keys():
        try:
            a,b=maps[0][oid],maps[1][oid]
            position=(1-interpolation['alpha'])*np.asarray(a['position'])+interpolation['alpha']*np.asarray(b['position'])
            radius=max(np.linalg.norm(a['model_bounds_min']),np.linalg.norm(a['model_bounds_max']),
                       np.linalg.norm(b['model_bounds_min']),np.linalg.norm(b['model_bounds_max']))
            center=inverse@(position-origin)
            lower_bound=center-radius*inverse_norm;upper_bound=center+radius*inverse_norm
            if upper_bound[2]<=.05:continue
            if lower_bound[2]>.05:
                ratios=np.array([lower_bound[:2]/lower_bound[2],lower_bound[:2]/upper_bound[2],
                                 upper_bound[:2]/lower_bound[2],upper_bound[:2]/upper_bound[2]])+.5
                if np.any(ratios.max(axis=0)<=0) or np.any(ratios.min(axis=0)>=1):continue
            obj = interpolate_object(maps[0][oid],maps[1][oid],interpolation['alpha'])
            status = geometry_evidence(obj,origin,basis)
            if status == 'CAMERA_INSIDE_OBB':
                near_camera.append(oid)
            if status != 'PROJECTABLE_OBB':
                continue
            bbox = project_box(obj,origin,basis,*manifest['resolution'])
            if bbox is None or min(bbox[2]-bbox[0],bbox[3]-bbox[1]) < 24:
                continue
            # Occupancy flags are used for review routing, not invisibility claims.
            subject = cameras[0].get('subject_observation') or {}
            appearance = subject.get('appearance') if str(subject.get('dataset_entity_id')) == oid else None
            candidates[oid] = {'bbox':bbox,'object':obj,'appearance':appearance,
                               'occupied':obj.get('current_vehicle_network_id') not in (None,0,'0')}
        except (TypeError,KeyError,ValueError):
            continue
    return {'candidates':candidates,'near_camera_entities':near_camera,
            'camera_origin':origin.tolist(),'ray_basis':basis.tolist()}


def qualify_windows(records, manifest, rows, target_frames=120):
    lookup = {r['proxy_frame_index']:r for r in rows}
    evidence = {}
    errors = {}
    windows = []
    for row in rows:
        i = row['proxy_frame_index']
        try:
            evidence[i] = frame_evidence(records,manifest,row)
        except (ValueError,TypeError,KeyError,AssertionError,np.linalg.LinAlgError) as error:
            errors[i] = str(error)
    for start in range(target_frames, max(lookup,default=-1)+1, target_frames):
        reference = start-target_frames
        indices = [reference,*range(start,start+target_frames)]
        result = {'reference_frame':reference,'target_start_frame':start,
                  'target_end_frame_exclusive':start+target_frames,'status':'REJECTED','reasons':[]}
        if any(i not in lookup for i in indices):
            result['reasons'] = ['INCOMPLETE_WINDOW']; windows.append(result); continue
        invalid = [i for i in indices if i in errors]
        if invalid:
            result['reasons'] = sorted({errors[i] for i in invalid});windows.append(result);continue
        selected = [lookup[i] for i in indices]
        if any(evidence[i]['near_camera_entities'] for i in indices):
            result['reasons'] = ['NEAR_CAMERA_PROXY_REQUIRES_RECOMPILATION'];windows.append(result);continue
        if len({r['server_clock_domain'] for r in selected}) != 1 or len({r['client_clock_domain'] for r in selected}) != 1:
            result['reasons'] = ['CLOCK_DISCONTINUITY'];windows.append(result);continue
        if any(b['server_time_estimate_ns']<=a['server_time_estimate_ns'] for a,b in zip(selected,selected[1:])):
            result['reasons'] = ['NONMONOTONIC_TIME'];windows.append(result);continue
        if any(b['rgb_decode_index'] != a['rgb_decode_index']+1 for a,b in zip(selected[1:],selected[2:])):
            result['reasons'] = ['RGB_DECODE_GAP'];windows.append(result);continue
        common = set(evidence[reference]['candidates'])
        for i in indices[1:]:
            common.intersection_update(evidence[i]['candidates'])
        objects = []
        for oid in sorted(common):
            observations = [evidence[i]['candidates'][oid] for i in indices]
            try:
                life = records.entity(oid)
                ticks = [r['world_capture_tick'] for r in selected]
                if life.get('spawn_sample_seq') is None or min(ticks)<life['spawn_sample_seq'] or (life.get('despawn_sample_seq') is not None and max(ticks)>=life['despawn_sample_seq']):
                    continue
                if any(model(o['object']['model_hash']) != model(life['model_hash']) for o in observations):
                    continue
            except (ValueError,KeyError):
                continue
            if any(o['occupied'] for o in observations):
                continue
            appearances=[o['appearance'] for o in observations]
            known=[a for a in appearances if a and a.get('available')]
            if known:
                signatures={json.dumps({k:a.get(k) for k in ('model_hash','components','props')},sort_keys=True) for a in known}
                if len(signatures)>1:continue
            # Identity is not state equality. Preserve observed state per frame.
            objects.append({'entity_id':oid,'model_hash':life['model_hash'],
                            'category':life.get('entity_type'), 'lifecycle':life,
                            'reference_bbox':observations[0]['bbox'],
                            'target_bboxes':[o['bbox'] for o in observations[1:]],
                            'appearance_reference_scope':'visible exterior only',
                            'rgb_visibility':'REQUIRES_VISUAL_REVIEW'})
        if not objects:
            result['reasons'] = ['NO_CONTINUOUS_RESOLVABLE_REFERENCE_OBJECT']; windows.append(result);continue
        result.update(status='NEEDS_VISUAL_REVIEW',objects=objects,
                      observed_at_ns=selected[0]['server_time_estimate_ns'],
                      used_at_ns=selected[1]['server_time_estimate_ns'],
                      target_end_ns=selected[-1]['server_time_estimate_ns'],
                      reference_rgb_decode_index=selected[0]['rgb_decode_index'],
                      target_rgb_start=selected[1]['rgb_decode_index'])
        windows.append(result)
    return windows, errors
