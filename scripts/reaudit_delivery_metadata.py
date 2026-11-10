"""Read-only full ZIP metadata audit; never executes packaged code or calls APIs.

All three primary manifests and every referenced sample are inspected. Media
presence/size and ZIP CRC+size duplicate candidates are NOT media decode, byte
identity proof, or visual alignment. Historical external paths are provenance,
not portable training dependencies. Output is restricted to the requested folder.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import io
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import statistics
import sys
import time
import unittest
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip')
DEFAULT_OUTPUT = ROOT / 'outputs/dataset_reaudit/metadata'
SPLITS = ('train', 'validation', 'test')
POINT = re.compile(r'^(ep_\d{8}_\d{6}_[^_]+)_(agent_\d+)_(segment_\d+)_[fw](\d+)$')
ARCHIVE = None
MEMBERS = None
RATE_TOLERANCE = .02
SIDECAR_METHOD = 'interpolated_recorded_sidecar_clock_at_decoded_rgb_pts_v1'


def strict_json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def relative_member(root, value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Expected nonempty POSIX relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or '..' in path.parts:
        raise ValueError('Absolute or escaping archive path')
    if str(path) != value or ':' in value or '\x00' in value:
        raise ValueError('Noncanonical archive path')
    return str(PurePosixPath(root) / path) if root else value


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value):
    return type(value) is int and value >= 0


def source_key(point_id):
    match = POINT.fullmatch(point_id or '')
    if not match:
        raise ValueError('Unrecognized source point identity: ' + str(point_id))
    return match.group(1, 2, 3), int(match.group(4))


def clock_rate(observations):
    """Compare published server elapsed against source RGB PTS, never CFR time.

    A 2% tolerance is a screening threshold, not an alignment acceptance limit.
    Integer origins are subtracted before conversion to seconds. Both endpoint
    ratio and least-squares slope are reported; no timestamps are repaired.
    """
    valid = [o for o in observations if integer(o.get('rgb_video_pts_ns'))
             and integer(o.get('server_time_estimate_ns'))]
    if len(valid) != len(observations) or len(valid) < 2:
        return dict(status='insufficient_numeric_evidence', observations=len(valid), flagged=None)
    valid = sorted(valid, key=lambda o: o['rgb_video_pts_ns'])
    first, last = valid[0], valid[-1]
    dx = last['rgb_video_pts_ns'] - first['rgb_video_pts_ns']
    dy = last['server_time_estimate_ns'] - first['server_time_estimate_ns']
    if dx <= 0:
        return dict(status='nonpositive_pts_span', observations=len(valid), flagged=None)
    x = [(o['rgb_video_pts_ns']-first['rgb_video_pts_ns'])/1e9 for o in valid]
    y = [(o['server_time_estimate_ns']-first['server_time_estimate_ns'])/1e9 for o in valid]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    variance = sum((v-mx)**2 for v in x)
    slope = sum((a-mx)*(b-my) for a,b in zip(x,y))/variance
    intercept = my-slope*mx
    residuals = [b-(intercept+slope*a) for a,b in zip(x,y)]
    ratio = dy/dx
    return dict(status='measured', observations=len(valid),
                first_source_frame=first.get('rgb_segment_frame_index',first.get('rgb_decode_index')),
                last_source_frame=last.get('rgb_segment_frame_index',last.get('rgb_decode_index')),
                first_rgb_decode_index=first.get('rgb_decode_index'), last_rgb_decode_index=last.get('rgb_decode_index'),
                first_rgb_pts_ns=first['rgb_video_pts_ns'], first_server_ns=first['server_time_estimate_ns'],
                last_rgb_pts_ns=last['rgb_video_pts_ns'], last_server_ns=last['server_time_estimate_ns'],
                rgb_pts_elapsed_sec=dx/1e9, server_elapsed_sec=dy/1e9,
                server_over_rgb_pts_elapsed=ratio, fitted_server_over_rgb_pts_slope=slope,
                endpoint_drift_ms=(dy-dx)/1e6, fitted_residual_rmse_ms=1000*math.sqrt(statistics.fmean(v*v for v in residuals)),
                fitted_residual_max_abs_ms=1000*max(abs(v) for v in residuals),
                tolerance_fraction=RATE_TOLERANCE,
                flagged=abs(ratio-1)>RATE_TOLERANCE or abs(slope-1)>RATE_TOLERANCE)


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    def quantile(q):
        index = q*(len(values)-1)
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return dict(count=len(values), minimum=values[0], p01=quantile(.01),
                median=quantile(.5), p95=quantile(.95), p99=quantile(.99), maximum=values[-1])


def overlapping_pairs(windows):
    """All overlapping half-open integer-frame windows within one source stream."""
    active = []
    for current in sorted(windows, key=lambda x: (x['start'], x['end'], x['record_index'])):
        active = [previous for previous in active if previous['end'] > current['start']]
        for previous in active:
            if previous['record_index'] != current['record_index']:
                yield previous, current, min(previous['end'], current['end']) - current['start']
        active.append(current)


def init_worker(archive):
    global ARCHIVE, MEMBERS
    ARCHIVE = zipfile.ZipFile(archive, 'r')
    MEMBERS = {info.filename: info for info in ARCHIVE.infolist()}


class RecordAudit:
    def __init__(self, row):
        self.row = row
        self.root = relative_member('', row['sample_root'])
        self.issues = {}
        self.checked_paths = set()
        self.json_cache = {}
        self.json_bytes = 0
        self.metrics = Counter()
        self.media = []

    def issue(self, code, detail, severity='error'):
        entry = self.issues.setdefault((severity, code), dict(severity=severity, code=code,
                                                            occurrences=0, examples=[]))
        entry['occurrences'] += 1
        if len(entry['examples']) < 5:
            entry['examples'].append(detail)

    def check(self, condition, code, detail, severity='error'):
        if not condition:
            self.issue(code, detail, severity)
        return bool(condition)

    def path(self, value, field):
        try:
            member = relative_member(self.root, value)
        except ValueError as error:
            self.issue('invalid_dependency_path', dict(field=field, value=value, reason=str(error)))
            return None
        self.checked_paths.add(member)
        info = MEMBERS.get(member)
        if info is None or info.is_dir():
            self.issue('missing_dependency', dict(field=field, member=member))
            return None
        if info.file_size == 0:
            self.issue('empty_dependency', dict(field=field, member=member))
        if member.lower().endswith(('.png', '.jpg', '.mp4', '.npy')):
            self.media.append(dict(field=field, member=member, size=info.file_size, crc=info.CRC))
        return member

    def read(self, value, field=None):
        member = self.path(value, field or value)
        if member is None:
            return None
        if member not in self.json_cache:
            try:
                raw = ARCHIVE.read(member)
                self.json_bytes += len(raw)
                self.json_cache[member] = strict_json(raw)
            except (ValueError, UnicodeError, zipfile.BadZipFile, RuntimeError, OSError) as error:
                self.issue('unreadable_json', dict(member=member, reason=str(error)))
                self.json_cache[member] = None
        return self.json_cache[member]

    def boxes(self, boxes, expected, field):
        valid = isinstance(boxes, list) and len(boxes) == expected
        if valid:
            valid = all(isinstance(b, list) and len(b) == 4 and all(number(v) for v in b)
                        and 0 <= b[0] < b[2] and 0 <= b[1] < b[3] for b in boxes)
        self.check(valid, 'invalid_bbox_metadata', dict(field=field, expected_count=expected))

    def joins(self, frames, cameras, worlds, episode, agent):
        if not isinstance(cameras, dict) or not isinstance(worlds, dict):
            self.issue('invalid_record_tables', 'Camera/world records must be dictionaries')
            return
        for frame in frames:
            index = frame.get('proxy_frame_index')
            interpolation = frame.get('camera_interpolation') or {}
            alpha = interpolation.get('alpha')
            lower, upper = (interpolation.get(k) for k in
                            ('lower_client_sample_seq', 'upper_client_sample_seq'))
            if not self.check(number(alpha) and 0 <= alpha <= 1 and integer(lower) and integer(upper),
                              'invalid_camera_interpolation', dict(frame=index, interpolation=interpolation)):
                continue
            selected = [cameras.get(str(key)) for key in (lower, upper)]
            if not self.check(all(isinstance(c, dict) for c in selected), 'missing_camera_join',
                              dict(frame=index, lower=lower, upper=upper)):
                continue
            self.metrics['camera_endpoint_joins'] += 2
            for key, c in zip((lower, upper), selected):
                self.check(c.get('episode_id') == episode and c.get('agent_id') == agent
                           and c.get('client_sample_seq') == key
                           and c.get('client_clock_domain') == frame.get('client_clock_domain')
                           and c.get('server_clock_domain') == frame.get('server_clock_domain'),
                           'camera_join_identity_or_clock_mismatch', dict(frame=index, key=key))
            times = [c.get('server_time_estimate_ns') for c in selected]
            if all(integer(t) for t in times):
                gap = times[1] - times[0]
                self.check(gap >= 0 and interpolation.get('gap_ns') == gap,
                           'camera_interpolation_gap_mismatch', dict(frame=index, actual=gap,
                                                                    declared=interpolation.get('gap_ns')))
                target_time = frame.get('server_time_estimate_ns')
                if integer(target_time):
                    delta = times[0] + round(alpha * gap) - target_time
                    self.metrics['interpolation_max_abs_error_ns'] = max(
                        self.metrics['interpolation_max_abs_error_ns'], abs(delta))
                    self.check(abs(delta) <= 2, 'camera_interpolation_time_mismatch',
                               dict(frame=index, reconstructed_minus_target_ns=delta))
            world_key = frame.get('world_capture_tick')
            world = worlds.get(str(world_key))
            if self.check(isinstance(world, dict), 'missing_world_join', dict(frame=index, key=world_key)):
                self.metrics['world_joins'] += 1
                self.check(world.get('episode_id') == episode and world.get('sample_seq') == world_key
                           and world.get('server_clock_domain') == frame.get('server_clock_domain'),
                           'world_join_identity_or_clock_mismatch', dict(frame=index, key=world_key))

    def run(self):
        row = self.row
        sample = self.read('sample.json')
        if not isinstance(sample, dict):
            raise ValueError('Missing or invalid sample.json object')
        sid, episode = row['sample_id'], row['episode_id']
        self.check(sample.get('sample_id') == sid and sample.get('episode_id') == episode,
                   'manifest_sample_identity_mismatch', dict(sample_id=sample.get('sample_id'),
                                                           episode_id=sample.get('episode_id')))
        self.check(self.root == 'samples/' + sid, 'sample_root_identity_mismatch', self.root)
        self.check(row.get('dataset') == 'data', 'unexpected_dataset_root', row.get('dataset'))
        self.check(sample.get('frame_count') == 120 and type(sample.get('frame_count')) is int
                   and sample.get('fps') == 24 and number(sample.get('fps')),
                   'invalid_target_format', dict(frame_count=sample.get('frame_count'), fps=sample.get('fps')))
        conditions, supervision = sample['conditions'], sample['supervision']
        for key in ('prompt', 'target_whitebox'):
            self.path(conditions.get(key), 'conditions.'+key)
        for key in ('target_rgb', 'valid_pixels'):
            self.path(supervision.get(key), 'supervision.'+key)
        prompt_path = self.path(conditions.get('prompt'), 'prompt')
        if prompt_path:
            try:
                self.check(bool(ARCHIVE.read(prompt_path).decode('utf-8-sig').strip()),
                           'empty_prompt', prompt_path)
            except (UnicodeError, zipfile.BadZipFile, RuntimeError, OSError) as error:
                self.issue('unreadable_prompt', str(error))
        pairs = conditions.get('reference_pairs') or [conditions.get('reference_pair')]
        pair_map = {}
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                raise ValueError('Invalid reference pair')
            rid = pair.get('reference_id', sid+':scene_ref')
            self.check(isinstance(rid, str) and bool(rid) and rid not in pair_map,
                       'duplicate_or_invalid_reference_id', rid)
            pair_map[rid] = pair
            for key in ('rgb', 'whitebox'):
                self.path(pair.get(key), f'reference[{i}].{key}')
        self.check(bool(pair_map), 'no_reference_pairs', sid)
        is_object = row['task'].startswith('object_')
        source, _ = source_key(sid)
        self.check(source[0] == episode, 'source_episode_mismatch', list(source))
        reference_sources = []
        if is_object:
            self.check(sample.get('agent_id') == source[1] and sample.get('source_segment') == source[2],
                       'object_source_mismatch', dict(source=list(source), agent=sample.get('agent_id'),
                                                      segment=sample.get('source_segment')))
            frames = self.read(sample['frame_state_index'])
            if not isinstance(frames, list) or len(frames) != 121:
                raise ValueError('Object timeline must be one reference plus 120 targets')
            reference, targets = frames[0], frames[1:]
            ref_time = reference.get('server_time_estimate_ns')
            ref_domain = reference.get('server_clock_domain')
            reference_sources.append(dict(source=list(source), frame=reference.get('rgb_decode_index'),
                                          time_ns=ref_time, server_clock_domain=ref_domain))
            self.check(len(pair_map) == 1, 'ambiguous_reference_source_mapping', len(pair_map))
            edges = sample.get('object_references', [])
            self.check(bool(edges), 'no_object_edges', sid)
            seen_edges = set()
            for j, edge in enumerate(edges):
                oid, rid = edge.get('object_id'), edge.get('reference_id')
                self.check((oid, rid) not in seen_edges, 'duplicate_object_reference_edge', dict(object_id=oid, reference_id=rid))
                seen_edges.add((oid, rid))
                self.check(rid in pair_map, 'unknown_object_reference_id', dict(edge=j, reference_id=rid))
                self.check(oid == episode+':entity:'+str(edge.get('dataset_entity_id')),
                           'object_identity_mismatch', dict(edge=j, object_id=oid))
                self.check(edge.get('used_by') == sid, 'object_used_by_mismatch', dict(edge=j, used_by=edge.get('used_by')))
                expected = [ref_time, targets[0].get('server_time_estimate_ns'), targets[-1].get('server_time_estimate_ns')]
                observed = [edge.get(key) for key in ('observed_at_ns', 'used_at_ns', 'used_until_ns')]
                self.check(observed == expected and all(integer(t) for t in observed),
                           'object_reference_time_mismatch', dict(edge=j, actual=observed, expected=expected))
                for key in ('reference_rgb', 'reference_whitebox', 'reference_crop', 'reference_mask', 'embedding'):
                    if key != 'embedding' or edge.get(key):
                        self.path(edge.get(key), f'object[{j}].{key}')
                for key in ('rgb', 'whitebox'):
                    self.check(edge.get('reference_'+key) == pair_map.get(rid, {}).get(key),
                               'object_reference_media_mismatch', dict(edge=j, kind=key))
                self.boxes([edge.get('reference_bbox_xyxy')], 1, f'object[{j}].reference_bbox')
                self.boxes(edge.get('target_projected_bboxes'), 120, f'object[{j}].target_bboxes')
            self.metrics['object_edges'] = len(edges)
            cameras, worlds = self.read(sample['camera_records']), self.read(sample['world_records'])
            self.joins(frames, cameras, worlds, episode, source[1])
            self.read(sample['review'])
        else:
            reference = self.read('reference_state.json')
            target = self.read('target_state.json')
            targets = self.read('target_frame_states.json')
            if not isinstance(reference, dict) or not isinstance(target, dict) or not isinstance(targets, list) or len(targets) != 120:
                raise ValueError('Scene needs reference/target states and 120 target frame states')
            frames = targets
            ref_time, ref_domain = reference.get('time_ns'), reference.get('server_clock_domain')
            ref_source, ref_index = source_key(reference['point_id'])
            reference_sources.append(dict(source=list(ref_source), frame=reference.get('rgb_decode_index'),
                                          time_ns=ref_time, server_clock_domain=ref_domain))
            for state, expected_source in ((reference, ref_source), (target, source)):
                actual_source = tuple(state.get(key) for key in ('episode_id', 'agent_id', 'segment'))
                self.check(actual_source == expected_source, 'state_source_identity_mismatch',
                           dict(point_id=state.get('point_id'), actual=actual_source, expected=expected_source))
            self.check(ref_source[0] == episode, 'reference_episode_mismatch', list(ref_source))
            self.check(reference.get('frame_index') == ref_index,
                       'reference_point_frame_mismatch', dict(point_frame=ref_index, state_frame=reference.get('frame_index')))
            self.check(target.get('point_id') == sid and sample['reference'].get('target_point') == sid
                       and sample['reference'].get('reference_point') == reference.get('point_id'),
                       'scene_reference_identity_mismatch', sample['reference'])
            self.check(sample.get('reference_time_ns') == ref_time, 'reference_time_mismatch',
                       dict(sample=sample.get('reference_time_ns'), state=ref_time))
            target_time = targets[0].get('server_time_estimate_ns')
            self.check(sample.get('target_time_ns') == target.get('time_ns') == target_time,
                       'target_start_time_mismatch', dict(sample=sample.get('target_time_ns'),
                                                       state=target.get('time_ns'), first_frame=target_time))
            self.check(target.get('server_clock_domain') == targets[0].get('server_clock_domain'),
                       'target_state_clock_mismatch', target.get('server_clock_domain'))
            self.check(target.get('rgb_decode_index') == targets[0].get('rgb_segment_frame_index')
                       and target.get('frame_index') == targets[0].get('proxy_frame_index'),
                       'target_state_frame_mismatch', dict(state_frame=target.get('frame_index'),
                                                          first_frame=targets[0].get('proxy_frame_index')))
            age = sample['reference'].get('age_sec')
            if integer(ref_time) and integer(target_time):
                self.check(number(age) and abs(age - (target_time-ref_time)/1e9) < 1e-6,
                           'reference_age_mismatch', dict(declared=age, derived=(target_time-ref_time)/1e9))
            self.metrics['scene_frames_without_packaged_camera_world_tables'] = len(targets)
        times = [f.get('server_time_estimate_ns') for f in targets]
        self.check(all(integer(t) for t in times), 'invalid_target_timestamps', times[:3])
        if not all(integer(t) for t in times):
            raise ValueError('Cannot audit noninteger timeline')
        domains = {f.get('server_clock_domain') for f in targets}
        clients = {f.get('client_clock_domain') for f in targets}
        self.check(len(domains) == 1 and None not in domains and '' not in domains,
                   'target_clock_domain_missing_or_changed', sorted(map(str, domains)))
        self.check(len(clients) == 1 and None not in clients and '' not in clients,
                   'target_client_domain_missing_or_changed', sorted(map(str, clients)))
        same_clock = ref_domain in domains and len(domains) == 1 and bool(ref_domain)
        self.check(same_clock, 'reference_target_clock_mismatch', dict(reference=ref_domain, target=list(domains)))
        earlier = integer(ref_time) and ref_time < times[0]
        self.check(earlier, 'reference_not_earlier_than_target', dict(reference_ns=ref_time, target_ns=times[0]))
        self.metrics['chronology_verified_same_server_clock'] = int(earlier and same_clock)
        differences = [b-a for a, b in zip(times, times[1:])]
        self.check(all(d > 0 for d in differences), 'nonmonotonic_target_time',
                   dict(indices=[i+1 for i, d in enumerate(differences) if d <= 0]))
        indices = {}
        for field in ('proxy_frame_index', 'rgb_decode_index', 'rgb_segment_frame_index', 'rgb_frame_index'):
            values = [f.get(field) for f in targets]
            indices[field] = values
            self.check(all(integer(v) for v in values), 'invalid_frame_indices', dict(field=field, first=values[:3]))
            if all(integer(v) for v in values):
                self.check(all(b == a+1 for a, b in zip(values, values[1:])),
                           'noncontiguous_target_frames', dict(field=field, start=values[0], end=values[-1]))
        for i, frame in enumerate(frames):
            detail = dict(frame=frame.get('proxy_frame_index'), timeline_index=i)
            provenance = frame.get('source_timestamp_provenance') or {}
            method = provenance.get('method')
            self.metrics['timestamp_provenance_method:'+str(method)] += 1
            self.check(frame.get('proxy_frame_index') == frame.get('rgb_segment_frame_index'),
                       'local_frame_mapping_mismatch', detail)
            # The sidecar schema retains global decoder indices; the descriptor
            # schema stores segment-local decoder indices. Do not mix domains.
            expected_decode = frame.get('rgb_frame_index') if method == SIDECAR_METHOD else frame.get('rgb_segment_frame_index')
            self.check(frame.get('rgb_decode_index') == expected_decode,
                       'decoder_frame_mapping_mismatch', dict(**detail, method=method,
                                                              expected=expected_decode, actual=frame.get('rgb_decode_index')))
            if 'original_video_pts_ns' in provenance:
                self.check(provenance['original_video_pts_ns'] == frame.get('rgb_video_pts_ns')
                           and integer(frame.get('rgb_video_pts_ns')), 'source_pts_provenance_mismatch', detail)
            elif method == SIDECAR_METHOD:
                lo, hi, fraction = (provenance.get(key) for key in
                                    ('lower_recorded_pts_ns','upper_recorded_pts_ns','fraction'))
                pts = frame.get('rgb_video_pts_ns')
                lower_index,upper_index = (provenance.get(key) for key in
                                           ('lower_recorded_frame_index','upper_recorded_frame_index'))
                valid = (integer(lo) and integer(hi) and lo < hi and integer(pts) and lo <= pts <= hi
                         and number(fraction) and 0 <= fraction <= 1
                         and integer(lower_index) and integer(upper_index) and lower_index < upper_index
                         and provenance.get('valid') is True)
                self.check(valid, 'invalid_sidecar_interpolation_metadata', dict(**detail, provenance=provenance))
                if valid:
                    self.check(abs((pts-lo)-fraction*(hi-lo)) <= 1,
                               'sidecar_pts_interpolation_mismatch', detail)
                timestamp_file = provenance.get('original_timestamp_file')
                if timestamp_file:
                    normalized = timestamp_file.replace('\\','/')
                    self.check('/'+source[0]+'/' in normalized and '/'+source[1]+'/' in normalized,
                               'source_timestamp_file_identity_mismatch', dict(**detail,path=timestamp_file))
                else:
                    self.issue('source_timestamp_file_unavailable', detail, 'warning')
            else:
                self.issue('source_pts_provenance_unavailable', dict(**detail,method=method), 'warning')
            descriptor = provenance.get('descriptor', '').replace('\\', '/')
            if descriptor:
                self.check('/'+source[1]+'/' in descriptor and descriptor.endswith('/'+source[2]+'.json'),
                           'source_descriptor_identity_mismatch', dict(**detail, descriptor=descriptor))
            elif method != SIDECAR_METHOD:
                self.issue('source_descriptor_unavailable', dict(**detail,method=method), 'warning')
            target_time = frame.get('server_time_estimate_ns')
            observation_time = frame.get('client_observation_server_time_ns')
            delta = frame.get('client_alignment_delta_ns')
            if integer(target_time) and integer(observation_time) and type(delta) is int:
                self.check(observation_time-target_time == delta, 'client_alignment_delta_mismatch', detail)
                self.metrics['max_abs_client_alignment_delta_ns'] = max(self.metrics['max_abs_client_alignment_delta_ns'], abs(delta))
            else:
                self.issue('missing_alignment_delta_evidence', detail)
            self.metrics['alignment_status:'+str(frame.get('status'))] += 1
        for field in ('rgb_video_pts_ns', 'rgb_segment_pts_ns', 'rgb_encoded_pts_ns'):
            values = [f.get(field) for f in targets]
            self.check(all(integer(v) for v in values), 'invalid_pts', dict(field=field))
            if all(integer(v) for v in values):
                self.check(all(b > a for a, b in zip(values, values[1:])),
                           'nonmonotonic_pts', dict(field=field))
        if max(differences) > 2e9 / 24:
            self.issue('capture_time_gap_over_two_output_frames',
                       dict(max_gap_ns=max(differences), indices=[i+1 for i,d in enumerate(differences) if d > 2e9/24]), 'warning')
        rates = dict(target_window=clock_rate(targets))
        if is_object:
            rates['reference_through_target'] = clock_rate(frames)
        for scope,rate in rates.items():
            if rate['flagged']:
                self.issue('server_rgb_pts_rate_outside_2_percent', dict(scope=scope, **rate), 'warning')
            elif rate['flagged'] is None:
                self.issue('server_rgb_pts_rate_unverifiable', dict(scope=scope, **rate))
        timing_observations = [dict(rgb_decode_index=f.get('rgb_decode_index'),
                                    rgb_segment_frame_index=f.get('rgb_segment_frame_index'),
                                    rgb_video_pts_ns=f.get('rgb_video_pts_ns'),
                                    server_time_estimate_ns=f.get('server_time_estimate_ns'),
                                    server_clock_domain=f.get('server_clock_domain')) for f in frames]
        # Parse additional sample-local JSON as metadata, without executing it.
        for name in self.row['_json_members']:
            self.read(name[len(self.root)+1:])
        output = dict(record_index=row['_record_index'], sample_id=sid, split=row['_manifest_split'],
                      task=row['task'], episode_id=episode, source=list(source),
                      reference_sources=reference_sources, target_count=len(targets),
                      target_start=indices['rgb_segment_frame_index'][0],
                      target_end_exclusive=indices['rgb_segment_frame_index'][-1]+1,
                      start_ns=times[0], end_ns=times[-1], server_clock_domain=targets[0].get('server_clock_domain'),
                      reference_age_sec=(times[0]-ref_time)/1e9 if integer(ref_time) and same_clock else None,
                      capture_span_sec=(times[-1]-times[0])/1e9,
                      max_capture_gap_ms=max(differences)/1e6,
                      clock_rates=rates, timing_observations=timing_observations,
                      frame_mapping=dict(output_index=list(range(120)), output_fps=24,
                                         **indices, server_time_estimate_ns=times,
                                         rgb_video_pts_ns=[f.get('rgb_video_pts_ns') for f in targets],
                                         rgb_segment_pts_ns=[f.get('rgb_segment_pts_ns') for f in targets],
                                         rgb_encoded_pts_ns=[f.get('rgb_encoded_pts_ns') for f in targets]),
                      metrics=dict(self.metrics), metadata_json_count=len(self.json_cache),
                      metadata_uncompressed_bytes=self.json_bytes, dependency_count=len(self.checked_paths),
                      media=list({(v['field'],v['member']):v for v in self.media}.values()),
                      issues=list(self.issues.values()), audit_completed=True)
        return output


def audit_record(row):
    audit = None
    try:
        audit = RecordAudit(row)
        return audit.run()
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        issues = list(audit.issues.values()) if audit else []
        issues.append(dict(severity='error', code='record_schema_audit_incomplete', occurrences=1,
                           examples=[type(error).__name__+': '+str(error)]))
        return dict(record_index=row['_record_index'], sample_id=row.get('sample_id'),
                    split=row['_manifest_split'], task=row.get('task'), episode_id=row.get('episode_id'),
                    issues=issues, audit_completed=False,
                    metadata_json_count=len(audit.json_cache) if audit else 0,
                    metadata_uncompressed_bytes=audit.json_bytes if audit else 0,
                    dependency_count=len(audit.checked_paths) if audit else 0, metrics={}, media=[])


class AuditTests(unittest.TestCase):
    def test_rate_catches_monotonic_slow_clock(self):
        origin = 4745898433600
        observations = [dict(rgb_decode_index=i, rgb_video_pts_ns=i*1000000000,
                             server_time_estimate_ns=origin+i*905000000) for i in range(56)]
        rate = clock_rate(observations)
        self.assertTrue(rate['flagged'])
        self.assertAlmostEqual(rate['server_over_rgb_pts_elapsed'], .905)
        self.assertAlmostEqual(rate['fitted_server_over_rgb_pts_slope'], .905)
        self.assertAlmostEqual(rate['endpoint_drift_ms'], -5225.)
        for o in observations:
            o['server_time_estimate_ns'] = origin+o['rgb_video_pts_ns']
        self.assertFalse(clock_rate(observations)['flagged'])

    def test_scene_fixture_and_injected_anomalies(self):
        global ARCHIVE, MEMBERS
        original = ARCHIVE, MEMBERS
        sid = 'ep_20260918_132805_7503_agent_000_segment_00000_f000120'
        rid = sid.replace('f000120','f000000')
        source, _ = source_key(sid)
        episode, agent, segment = source
        root = 'samples/'+sid
        def build(rate=1., future=False, missing=False, sidecar=False,
                  bad_fraction=False, wrong_local=False, missing_descriptor=False):
            frames = []
            for i in range(120,240):
                pts = i*41666667
                t = 100000000000+round(pts*rate)
                frames.append(dict(proxy_frame_index=i,rgb_decode_index=i,rgb_segment_frame_index=i,
                                   rgb_frame_index=1440+i,rgb_video_pts_ns=pts,rgb_segment_pts_ns=pts,
                                   rgb_encoded_pts_ns=pts,server_time_estimate_ns=t,
                                   server_clock_domain='server',client_clock_domain='client',
                                   client_observation_server_time_ns=t,client_alignment_delta_ns=0,
                                   source_timestamp_provenance=dict(original_video_pts_ns=pts,
                                       descriptor=f'/source/agents/{agent}/{segment}.json')))
                if sidecar:
                    frames[-1]['rgb_decode_index'] = 1440+i
                    frames[-1]['source_timestamp_provenance'] = dict(
                        method=SIDECAR_METHOD, lower_recorded_pts_ns=pts-10,
                        upper_recorded_pts_ns=pts+10, fraction=.2 if bad_fraction else .5,
                        lower_recorded_frame_index=1440+i,upper_recorded_frame_index=1441+i,
                        valid=True,original_timestamp_file=f'/source/{episode}/agents/{agent}/frame_timestamps.jsonl')
                if wrong_local:
                    frames[-1]['rgb_segment_frame_index'] += 1
                if missing_descriptor:
                    frames[-1]['source_timestamp_provenance'].pop('descriptor',None)
            ref_time = frames[0]['server_time_estimate_ns']+1 if future else 100000000000
            sample = dict(sample_id=sid,episode_id=episode,frame_count=120,fps=24,
                          reference_time_ns=ref_time,target_time_ns=frames[0]['server_time_estimate_ns'],
                          reference=dict(reference_point=rid,target_point=sid,
                                         age_sec=(frames[0]['server_time_estimate_ns']-ref_time)/1e9),
                          conditions=dict(prompt='prompt.txt',target_whitebox='target_whitebox.mp4',
                                          reference_pair=dict(rgb='ref.png',whitebox='ref_w.png')),
                          supervision=dict(target_rgb='target_rgb.mp4',valid_pixels='valid.png'))
            ref = dict(point_id=rid,episode_id=episode,agent_id=agent,segment=segment,
                       frame_index=0,rgb_decode_index=0,time_ns=ref_time,server_clock_domain='server')
            target = dict(ref,point_id=sid,frame_index=120,rgb_decode_index=120,
                          time_ns=frames[0]['server_time_estimate_ns'])
            stream = io.BytesIO()
            with zipfile.ZipFile(stream,'w') as z:
                for name,data in [('sample.json',sample),('reference_state.json',ref),
                                  ('target_state.json',target),('target_frame_states.json',frames)]:
                    z.writestr(root+'/'+name,json.dumps(data))
                for name in ('prompt.txt','target_whitebox.mp4','target_rgb.mp4','ref.png','ref_w.png','valid.png'):
                    if not (missing and name=='ref.png'):
                        z.writestr(root+'/'+name,b'fixture')
            stream.seek(0)
            return stream
        row = dict(sample_id=sid,episode_id=episode,sample_root=root,dataset='data',task='scene_background',
                   _record_index=0,_manifest_split='train',_json_members=[])
        try:
            for kwargs,expected in [({},None),({'rate':.905},'server_rgb_pts_rate_outside_2_percent'),
                                    ({'future':True},'reference_not_earlier_than_target'),
                                    ({'missing':True},'missing_dependency'), ({'sidecar':True},None),
                                    ({'sidecar':True,'bad_fraction':True},'sidecar_pts_interpolation_mismatch'),
                                    ({'sidecar':True,'wrong_local':True},'local_frame_mapping_mismatch'),
                                    ({'missing_descriptor':True},'source_descriptor_unavailable')]:
                with zipfile.ZipFile(build(**kwargs),'r') as z:
                    ARCHIVE, MEMBERS = z, {i.filename:i for i in z.infolist()}
                    result = audit_record(row)
                    self.assertTrue(result['audit_completed'])
                    codes = {i['code'] for i in result['issues']}
                    if expected:
                        self.assertIn(expected,codes)
                        if expected=='source_descriptor_unavailable':
                            self.assertNotIn('source_descriptor_identity_mismatch',codes)
                    else:
                        self.assertFalse(codes)
        finally:
            ARCHIVE, MEMBERS = original
    def test_archive_paths(self):
        self.assertEqual(relative_member('samples/a', 'references/rgb.png'), 'samples/a/references/rgb.png')
        for value in ('../x', '/x', 'C:/x', 'C:x', 'a\\x', 'a/../x', 'a//x', ''):
            with self.assertRaises(ValueError):
                relative_member('samples/a', value)

    def test_strict_json(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                strict_json(raw)
        self.assertEqual(strict_json('{"t":4745898433600}')['t'], 4745898433600)

    def test_overlapping_windows(self):
        rows = [dict(record_index=i, start=a, end=b) for i,(a,b) in
                enumerate(((0,120),(120,240),(119,239),(0,120)))]
        pairs = {(min(a['record_index'],b['record_index']),max(a['record_index'],b['record_index'])): n
                 for a,b,n in overlapping_pairs(rows)}
        self.assertNotIn((0,1), pairs)
        self.assertEqual(pairs[(0,2)], 1)
        self.assertEqual(pairs[(1,2)], 119)
        self.assertEqual(pairs[(0,3)], 120)


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def finalize_cross_checks(output):
    """Independent joins over audited records, without rereading heavy JSON.

    This can also be rerun on a completed audit: its own findings are replaced,
    never duplicated. Source clock joins deliberately ignore episode labels.
    """
    summary = strict_json((output/'summary.json').read_bytes())
    records = [strict_json(line) for line in (output/'records.jsonl').read_bytes().splitlines()]
    fingerprints, clocks = defaultdict(list), defaultdict(list)
    aliases, paired_media_candidates = [], []
    for r in records:
        media = r['media']
        target = next((m for m in media if m['field']=='supervision.target_rgb'), None)
        whitebox = next((m for m in media if m['field']=='conditions.target_whitebox'), None)
        inputs = [m for m in media if m['field'].startswith('reference[') or m['field']=='conditions.target_whitebox']
        if target:
            for condition in inputs:
                if condition['member']==target['member']:
                    aliases.append(dict(sample_id=r['sample_id'], split=r['split'], input_field=condition['field'], member=target['member']))
            if whitebox and (target['size'],target['crc'])==(whitebox['size'],whitebox['crc']):
                paired_media_candidates.append(dict(sample_id=r['sample_id'], split=r['split'],
                                                    rgb=target['member'], whitebox=whitebox['member']))
        for m in media:
            if (m['field'].startswith('reference[') and m['field'].endswith('.rgb')) or m['field']=='supervision.target_rgb':
                key = (PurePosixPath(m['member']).suffix.lower(),m['size'],m['crc'])
                fingerprints[key].append(dict(sample_id=r['sample_id'], split=r['split'], field=m['field'], member=m['member']))
        if r['audit_completed'] and r['server_clock_domain']:
            key = (r['server_clock_domain'],r['source'][1])
            clocks[key].append(dict(record_index=r['record_index'], sample_id=r['sample_id'], split=r['split'],
                                    episode_id=r['episode_id'],start=r['start_ns'],end=r['end_ns']+1))
    cross_media = [dict(extension=k[0],size=k[1],crc32=k[2],members=v) for k,v in fingerprints.items()
                   if len({m['split'] for m in v})>1]
    cross_clock = []
    for key,values in clocks.items():
        # Within-split overlap is already reported by the frame-window audit.
        if len({v['split'] for v in values}) < 2:
            continue
        for left,right,overlap in overlapping_pairs(values):
            if left['split'] != right['split']:
                cross_clock.append(dict(server_clock_domain=key[0],agent_id=key[1],
                                        left=left,right=right,overlap_ns=overlap))
    checks = dict(records=len(records), rgb_media_references=sum(len(v) for v in fingerprints.values()),
                  fingerprint_basis='ZIP central-directory CRC32 plus uncompressed size; candidates only, bytes not reread',
                  cross_split_rgb_media_candidates=cross_media,
                  cross_split_clock_time_overlaps=cross_clock,
                  condition_target_rgb_path_aliases=aliases,
                  rgb_whitebox_identical_fingerprint_candidates=paired_media_candidates,
                  limits='No media decode, optical flow, visual matching or proof of byte identity.')
    dump_json(output/'cross_split_media_checks.json',checks)
    own_codes = {'cross_split_any_rgb_media_candidate','cross_split_same_clock_time_overlap',
                 'condition_target_rgb_path_alias','rgb_whitebox_same_fingerprint_candidate'}
    issues = [v for v in strict_json((output/'global_issues.json').read_bytes()) if v['code'] not in own_codes]
    for code,values,severity in [('cross_split_any_rgb_media_candidate',cross_media,'warning'),
                                 ('cross_split_same_clock_time_overlap',cross_clock,'error'),
                                 ('condition_target_rgb_path_alias',aliases,'error'),
                                 ('rgb_whitebox_same_fingerprint_candidate',paired_media_candidates,'warning')]:
        if values:
            issues.append(dict(code=code,severity=severity,detail=dict(count=len(values),evidence='cross_split_media_checks.json')))
    summary['additional_cross_checks'] = dict(rgb_media_references=checks['rgb_media_references'],
                                              cross_split_rgb_media_candidates=len(cross_media),
                                              cross_split_clock_time_overlaps=len(cross_clock),
                                              condition_target_rgb_path_aliases=len(aliases),
                                              rgb_whitebox_same_fingerprint_candidates=len(paired_media_candidates))
    summary['target_frame_mapping_rows'] = sum(r.get('target_count',0) for r in records)
    summary['global_issue_count'] = len(issues)
    summary['global_issue_counts'] = dict(Counter(v['code'] for v in issues))
    if any(v['severity']=='error' for v in issues) or summary['metadata_error_records']:
        summary['status'] = 'METADATA_FINDINGS'
    elif issues or summary['metadata_warning_records']:
        summary['status'] = 'METADATA_WARNINGS'
    dump_json(output/'global_issues.json',issues)
    dump_json(output/'summary.json',summary)
    marker = '\n## Additional Cross Checks\n'
    report = (output/'REPORT.md').read_text(encoding='utf-8').split(marker)[0]
    report += marker+'\n'+json.dumps(summary['additional_cross_checks'],indent=2)+'\n\n'
    report += 'These are metadata joins and duplicate candidates, not visual alignment checks.\n'
    (output/'REPORT.md').write_text(report,encoding='utf-8')
    if summary.get('evidence_reuse',{}).get('enabled'):
        write_schema_review(output,summary,records)
    return summary


def write_schema_review(output, summary, records):
    """Keep one raw schema comparison and the three concrete consequences."""
    history = strict_json((output/'schema_recheck_history.json').read_bytes())
    rechecked = set(history['reread_sample_ids'])
    example_mapping = None
    with (output/'frame_mappings.jsonl').open(encoding='utf-8') as source:
        for line in source:
            mapping = strict_json(line)
            if mapping['sample_id'] in rechecked and mapping['rgb_decode_index'][0] != mapping['rgb_segment_frame_index'][0]:
                example_mapping = mapping
                break
    if example_mapping is None:
        return
    sid = example_mapping['sample_id']
    root = 'samples/'+sid
    with zipfile.ZipFile(summary['archive'],'r') as z:
        target = strict_json(z.read(root+'/target_state.json'))
        frames = strict_json(z.read(root+'/target_frame_states.json'))
    frame = frames[0]
    state_keys = ('point_id','episode_id','agent_id','segment','frame_index','rgb_decode_index','time_ns','server_clock_domain')
    frame_keys = ('proxy_frame_index','rgb_segment_frame_index','rgb_decode_index','rgb_frame_index',
                  'rgb_video_pts_ns','rgb_segment_pts_ns','server_time_estimate_ns','server_clock_domain','source_timestamp_provenance')
    corrected = [r for r in records if r['sample_id'] in rechecked]
    evidence = dict(previous_error_records=history['previous_error_records'],
                    reread_records=len(corrected), current_error_records=summary['metadata_error_records'],
                    current_errors_in_rechecked_records=sum(any(i['severity']=='error' for i in r['issues']) for r in corrected),
                    sidecar_frames_validated=sum(r['metrics'].get('timestamp_provenance_method:'+SIDECAR_METHOD,0) for r in corrected),
                    raw_example=dict(sample_id=sid, target_state_member=root+'/target_state.json',
                                     frame_state_member=root+'/target_frame_states.json',
                                     target_state={k:target.get(k) for k in state_keys},
                                     first_target_frame={k:frame.get(k) for k in frame_keys}),
                    actual_impacts=[
                        dict(kind='frame_index_namespaces',previous_flagged_records=history['previous_issue_counts'].get('error:local_frame_mapping_mismatch',{}).get('records',0),
                             conclusion='Segment-local target_state indices match proxy_frame_index and rgb_segment_frame_index. Sidecar-schema rgb_decode_index matches global rgb_frame_index.',
                             impact='Use segment-local indices for segment media; global indices for original decoder streams. These differences do not prove a bad exported video or shifted frame.'),
                        dict(kind='optional_descriptor',previous_flagged_records=history['previous_issue_counts'].get('error:source_descriptor_identity_mismatch',{}).get('records',0),
                             conclusion='The sidecar schema supplies original_timestamp_file rather than descriptor. Absence of descriptor is not an identity mismatch.',
                             impact='Packaged training dependencies still exist. External sidecar content and original clock calibration remain unverified by this ZIP-only audit.'),
                        dict(kind='timestamp_provenance_schema',previous_flagged_records=history['previous_issue_counts'].get('error:source_pts_provenance_mismatch',{}).get('records',0),
                             conclusion='For sidecar schema, RGB PTS is validated against lower/upper recorded PTS and fraction, not a nonexistent original_video_pts_ns field.',
                             impact='Bracket validity and fraction arithmetic were checked for every reread frame; server/RGB PTS rate is an independent test and its warnings remain.')],
                    timing_findings_unchanged=dict(target_windows_flagged=summary['clock_rates']['target_windows_flagged'],
                                                    samples_including_reference_spans=summary['issue_counts']['warning:server_rgb_pts_rate_outside_2_percent']['records'],
                                                    source_segments_flagged=summary['clock_rates']['source_segments_flagged'],
                                                    capture_gap_samples=summary['issue_counts']['warning:capture_time_gap_over_two_output_frames']['records']),
                    evidence_reuse=summary['evidence_reuse'], visual_alignment_verified=False)
    dump_json(output/'schema_variant_review.json',evidence)
    marker = '\n## Schema Correction\n'
    text = (output/'REPORT.md').read_text(encoding='utf-8').split(marker)[0]
    text += marker+f"\nThe previous {history['previous_error_records']} errors were schema-assumption false positives, not established source contradictions. "
    text += f"Rechecked {len(corrected)} records ({summary['evidence_reuse']['freshly_read_metadata_bytes']:,} bytes); reused {summary['evidence_reuse']['cached_records']} completed checks. Current metadata errors: {summary['metadata_error_records']}.\n\n"
    text += f"Raw example: `{sid}`.\n\n| Field | target_state | first target frame |\n|---|---:|---:|\n"
    text += f"| Segment-local frame | {target['frame_index']} | {frame['rgb_segment_frame_index']} |\n"
    text += f"| rgb_decode_index | {target['rgb_decode_index']} (local) | {frame['rgb_decode_index']} (global) |\n"
    text += f"| RGB global frame | not supplied | {frame['rgb_frame_index']} |\n"
    text += f"| Server timestamp ns | {target['time_ns']} | {frame['server_time_estimate_ns']} |\n\n"
    for item in evidence['actual_impacts']:
        text += '- '+item['conclusion']+' '+item['impact']+'\n'
    text += '\nFull raw fields and correction history: schema_variant_review.json and schema_recheck_history.json. No archive or timestamps were changed.\n'
    (output/'REPORT.md').write_text(text,encoding='utf-8')


def restore_cached_record(result, mapping):
    """Restore only already-audited numeric timelines; no new provenance claims."""
    result = dict(result)
    result['frame_mapping'] = {k:v for k,v in mapping.items() if k not in ('sample_id','split','source')}
    observations = [dict(rgb_decode_index=decode,rgb_segment_frame_index=local,
                          rgb_video_pts_ns=pts,server_time_estimate_ns=timestamp,
                          server_clock_domain=result['server_clock_domain'])
                    for decode,local,pts,timestamp in zip(mapping['rgb_decode_index'],
                                                         mapping['rgb_segment_frame_index'],
                                                         mapping['rgb_video_pts_ns'],mapping['server_time_estimate_ns'])]
    if result['task'].startswith('object_'):
        reference = result['reference_sources'][0]
        rate = result['clock_rates']['reference_through_target']
        observations.insert(0,dict(rgb_decode_index=reference['frame'],rgb_segment_frame_index=reference['frame'],
                                   rgb_video_pts_ns=rate['first_rgb_pts_ns'],server_time_estimate_ns=reference['time_ns'],
                                   server_clock_domain=reference['server_clock_domain']))
    result['timing_observations'] = observations
    return result


def run(archive, output, workers, expected_records, recheck_existing=False):
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).isoformat()
    before = archive.stat()
    output.mkdir(parents=True, exist_ok=True)
    global_issues = []
    def finding(code, detail, severity='error'):
        global_issues.append(dict(code=code, severity=severity, detail=detail))
    with zipfile.ZipFile(archive, 'r') as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        members = {i.filename:i for i in infos}
        repeated_paths = [n for n,c in Counter(names).items() if c>1]
        if repeated_paths:
            finding('duplicate_archive_member_names', repeated_paths)
        for info in infos:
            try:
                relative_member('', info.filename.rstrip('/'))
            except ValueError as error:
                finding('unsafe_archive_member', dict(member=info.filename, reason=str(error)))
        by_root = defaultdict(list)
        for name in names:
            if name.startswith('samples/') and name.endswith('.json'):
                by_root[name.rsplit('/',1)[0]].append(name)
        rows, split_counts = [], {}
        for split in SPLITS:
            split_rows = []
            for line_number, line in enumerate(z.read(f'manifests/{split}.jsonl').splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = strict_json(line)
                    if not isinstance(row,dict):
                        raise ValueError('Manifest row must be object')
                    for key in ('sample_id','sample_root','episode_id','task','dataset','split'):
                        if not isinstance(row.get(key),str) or not row[key]:
                            raise ValueError('Missing/invalid '+key)
                except (ValueError, UnicodeError) as error:
                    finding('invalid_manifest_row', dict(split=split, line=line_number, error=str(error)))
                    continue
                if row['split'] != split:
                    finding('manifest_split_mismatch', dict(sample_id=row['sample_id'], manifest=split, declared=row['split']))
                row.update(_manifest_split=split, _line=line_number, _record_index=len(rows),
                           _json_members=by_root.get(row['sample_root'], []))
                rows.append(row)
                split_rows.append(row)
            split_counts[split] = len(split_rows)
        if len(rows) != expected_records:
            finding('unexpected_record_count', dict(expected=expected_records, actual=len(rows)))
        declared = strict_json(z.read('DATASET_REPORT.json'))
        for key, actual in [('records',len(rows)), ('splits',split_counts),
                            ('tasks',dict(Counter(r['task'] for r in rows)))]:
            if declared.get(key) != actual:
                finding('dataset_report_mismatch', dict(field=key, declared=declared.get(key), actual=actual))
        groups = {}
        for key in ('sample_id','sample_root','episode_id','balance_group'):
            grouped = defaultdict(list)
            for row in rows:
                if row.get(key) is not None:
                    grouped[row[key]].append(row)
            groups[key] = grouped
            for value, group in grouped.items():
                splits = sorted({r['_manifest_split'] for r in group})
                if len(splits)>1:
                    finding('cross_split_'+key, dict(value=value, splits=splits, sample_ids=[r['sample_id'] for r in group]))
                elif key in ('sample_id','sample_root') and len(group)>1:
                    finding('duplicate_'+key, dict(value=value, record_indices=[r['_record_index'] for r in group]))
        split_registry = strict_json(z.read('provenance/cumulative_audit/episode_splits.json'))
        for row in rows:
            if split_registry.get(row['episode_id']) != row['_manifest_split']:
                finding('episode_registry_mismatch', dict(sample_id=row['sample_id'], registry=split_registry.get(row['episode_id'])))
        archived_samples = {n[:-len('/sample.json')] for n in names if n.startswith('samples/') and n.endswith('/sample.json')}
        orphan = sorted(archived_samples-set(groups['sample_root']))
        if orphan:
            finding('unmanifested_archived_samples', orphan, 'warning')
        rounds = []
        train_by_id = {r['sample_id']:r for r in rows if r['_manifest_split']=='train'}
        for name in names:
            if not name.startswith('manifests/train_round_') or not name.endswith('.jsonl'):
                continue
            round_rows = [strict_json(line) for line in z.read(name).splitlines() if line.strip()]
            ids = [r['sample_id'] for r in round_rows]
            missing = [sid for sid in ids if sid not in train_by_id]
            repeated = [sid for sid,count in Counter(ids).items() if count>1]
            mismatched = [r['sample_id'] for r in round_rows if r['sample_id'] in train_by_id
                          and r != {k:v for k,v in train_by_id[r['sample_id']].items() if not k.startswith('_')}]
            if missing or repeated or mismatched:
                finding('invalid_training_round', dict(manifest=name, not_in_train=missing,
                                                       repeated=repeated, row_mismatch=mismatched))
            rounds.append(dict(manifest=name, records=len(ids), task_counts=dict(Counter(r['task'] for r in round_rows))))
    cached, previous_summary, recheck_ids = {}, None, []
    if recheck_existing:
        previous_summary = strict_json((output/'summary.json').read_bytes())
        if (previous_summary['archive_bytes'],previous_summary['archive_mtime_ns']) != (before.st_size,before.st_mtime_ns):
            raise ValueError('Archive size/mtime changed; cached audit cannot be reused')
        if previous_summary['completed_records'] != len(rows):
            raise ValueError('Cached audit must have completed every primary-manifest record')
        old_records = [strict_json(line) for line in (output/'records.jsonl').read_bytes().splitlines()]
        mappings = {m['sample_id']:m for m in map(strict_json,(output/'frame_mappings.jsonl').read_bytes().splitlines())}
        if len(old_records) != len(rows) or len(mappings) != len(rows):
            raise ValueError('Incomplete cached records or frame mappings')
        for row,old in zip(rows,old_records):
            if (row['sample_id'],row['_manifest_split'],row['episode_id'],row['task']) != (
                    old['sample_id'],old['split'],old['episode_id'],old['task']):
                raise ValueError('Manifest differs from cached audit')
            if any(i['severity']=='error' for i in old['issues']):
                recheck_ids.append(row['_record_index'])
            else:
                cached[row['_record_index']] = restore_cached_record(old,mappings[old['sample_id']])
        dump_json(output/'schema_recheck_history.json',dict(
            previous_completed_utc=previous_summary['finished_utc'],
            previous_elapsed_seconds=previous_summary['elapsed_seconds'],
            previous_error_records=previous_summary['metadata_error_records'],
            previous_issue_counts=previous_summary['issue_counts'],
            cached_records=len(cached), reread_records=len(recheck_ids),
            reread_sample_ids=[rows[i]['sample_id'] for i in recheck_ids],
            reason='Explicit schema correction: sidecar interpolation and global decoder indices; unchanged rows reuse their completed archive checks.'))
        del mappings,old_records
    fresh_rows = [row for row in rows if row['_record_index'] not in cached]
    print(json.dumps(dict(stage='audit_start', records=len(rows), splits=split_counts, workers=workers,
                          cached_records=len(cached), fresh_records=len(fresh_rows))), flush=True)
    results, issue_counts, issue_records, metric_totals = [], Counter(), defaultdict(set), Counter()
    source_timing = defaultdict(dict)
    source_records = defaultdict(set)
    json_count = json_bytes = dependency_count = 0
    with (output/'records.jsonl').open('w',encoding='utf-8') as records_out, \
         (output/'frame_mappings.jsonl').open('w',encoding='utf-8') as mapping_out, \
         (output/'anomalies.jsonl').open('w',encoding='utf-8') as anomalies_out, \
         ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(str(archive),)) as pool:
        fresh = pool.map(audit_record, fresh_rows, chunksize=8, buffersize=workers*2)
        ordered = (cached.pop(row['_record_index']) if row['_record_index'] in cached else next(fresh) for row in rows)
        for result in ordered:
            observations = result.pop('timing_observations', [])
            for observation in observations:
                key = (*result['source'], observation['server_clock_domain'])
                frame = observation.get('rgb_segment_frame_index',observation['rgb_decode_index'])
                previous = source_timing[key].get(frame)
                if previous is not None and previous != observation:
                    finding('shared_source_frame_time_conflict', dict(source=list(key), frame=frame,
                                                                     sample_id=result['sample_id'],
                                                                     previous=previous, actual=observation))
                else:
                    source_timing[key][frame] = observation
                source_records[key].add(result['record_index'])
            mapping = result.pop('frame_mapping', None)
            if mapping:
                mapping_out.write(json.dumps(dict(sample_id=result['sample_id'], split=result['split'],
                                                  source=result['source'], **mapping), separators=(',',':'))+'\n')
            for issue in result['issues']:
                issue_counts[issue['severity']+':'+issue['code']] += issue['occurrences']
                issue_records[issue['severity']+':'+issue['code']].add(result['record_index'])
                anomalies_out.write(json.dumps(dict(sample_id=result['sample_id'], split=result['split'],
                                                    task=result['task'], **issue), ensure_ascii=False)+'\n')
            for key,value in result['metrics'].items():
                if 'max_' in key:
                    metric_totals[key] = max(metric_totals[key],value)
                else:
                    metric_totals[key] += value
            json_count += result['metadata_json_count']
            json_bytes += result['metadata_uncompressed_bytes']
            dependency_count += result['dependency_count']
            records_out.write(json.dumps(result, separators=(',',':'), ensure_ascii=False)+'\n')
            results.append(result)
            if len(results)%100 == 0:
                records_out.flush(); mapping_out.flush(); anomalies_out.flush()
                print(json.dumps(dict(stage='records', processed=len(results), total=len(rows),
                                      elapsed_sec=round(time.monotonic()-started,1),
                                      error_records=sum(any(i['severity']=='error' for i in r['issues']) for r in results))), flush=True)
    source_rates = []
    for key,frame_map in sorted(source_timing.items()):
        ids = sorted(source_records[key])
        rate = clock_rate(list(frame_map.values()))
        source_rates.append(dict(source=list(key[:3]), server_clock_domain=key[3],
                                 record_indices=ids, sample_ids=[rows[i]['sample_id'] for i in ids],
                                 splits=sorted({rows[i]['_manifest_split'] for i in ids}),
                                 baseline_scope='earliest to latest packaged frame with source RGB PTS; not necessarily source frame zero',
                                 **rate))
    flagged_sources = [r for r in source_rates if r['flagged']]
    affected_by_source_rate = set(i for r in flagged_sources for i in r['record_indices'])
    if flagged_sources:
        finding('source_server_rgb_pts_rate_outside_2_percent',
                dict(source_segments=len(flagged_sources), affected_records=len(affected_by_source_rate)), 'warning')
    target_rates = [dict(sample_id=r['sample_id'], split=r['split'], task=r['task'], episode_id=r['episode_id'],
                         **r['clock_rates']['target_window']) for r in results if r['audit_completed']]
    rate_report = dict(definition='Published server elapsed / source RGB PTS elapsed; source CFR output time is not used as capture evidence.',
                       tolerance_fraction=RATE_TOLERANCE, action='flag_only_no_timestamp_changes',
                       target_windows=distribution([r['server_over_rgb_pts_elapsed'] for r in target_rates if r['status']=='measured']),
                       source_segments=distribution([r['server_over_rgb_pts_elapsed'] for r in source_rates if r['status']=='measured']),
                       target_windows_flagged=sum(bool(r['flagged']) for r in target_rates),
                       source_segments_flagged=len(flagged_sources), records_in_flagged_source_segments=len(affected_by_source_rate),
                       by_split={split:dict(ratios=distribution([r['server_over_rgb_pts_elapsed'] for r in target_rates if r['split']==split and r['status']=='measured']),
                                           flagged=sum(bool(r['flagged']) for r in target_rates if r['split']==split)) for split in SPLITS},
                       by_episode={episode:dict(ratios=distribution([r['server_over_rgb_pts_elapsed'] for r in target_rates if r['episode_id']==episode and r['status']=='measured']),
                                               flagged=sum(bool(r['flagged']) for r in target_rates if r['episode_id']==episode)) for episode in groups['episode_id']},
                       worst_target_windows=sorted([r for r in target_rates if r['status']=='measured'],
                                                   key=lambda r:abs(r['server_over_rgb_pts_elapsed']-1), reverse=True)[:30],
                       worst_source_segments=sorted([r for r in source_rates if r['status']=='measured'],
                                                    key=lambda r:abs(r['server_over_rgb_pts_elapsed']-1), reverse=True)[:30],
                       limitations=['2 percent is an audit screening threshold, not proof of visual synchronization.',
                                    'Scene historical references lack RGB PTS in reference_state; their rate is not inferred from other cameras.',
                                    'An affine rate near one can still have a constant timestamp offset.',
                                    'No optical flow evidence was measured or adopted in this metadata-only audit.'])
    dump_json(output/'clock_rate_summary.json',rate_report)
    with (output/'source_clock_rates.jsonl').open('w',encoding='utf-8') as dest:
        for value in source_rates:
            dest.write(json.dumps(value)+'\n')
    del source_timing
    windows, exact, content = defaultdict(list), defaultdict(list), defaultdict(list)
    reference_uses = defaultdict(list)
    for result in results:
        if not result['audit_completed']:
            continue
        key = tuple(result['source'])
        window = dict(record_index=result['record_index'], sample_id=result['sample_id'], split=result['split'],
                      task=result['task'], start=result['target_start'], end=result['target_end_exclusive'])
        windows[key].append(window)
        exact[(*key,window['start'],window['end'])].append(window)
        for ref in result['reference_sources']:
            reference_uses[tuple(ref['source'])].append(dict(**window, reference_frame=ref['frame'], reference_time_ns=ref['time_ns']))
        target_rgb = next((m for m in result['media'] if m['field']=='supervision.target_rgb'), None)
        if target_rgb:
            content[(target_rgb['size'],target_rgb['crc'])].append(window)
    duplicate_groups = [dict(source_window=list(key), records=values) for key,values in exact.items() if len(values)>1]
    content_candidates = [dict(size=key[0], crc32=key[1], records=values,
                               cross_split=len({v['split'] for v in values})>1)
                          for key,values in content.items() if len(values)>1]
    overlap_counts = Counter()
    with (output/'source_window_overlaps.jsonl').open('w',encoding='utf-8') as dest:
        for source,values in windows.items():
            for left,right,count in overlapping_pairs(values):
                cross = left['split'] != right['split']
                identical = left['start']==right['start'] and left['end']==right['end']
                overlap_counts['cross_split' if cross else 'within_split'] += 1
                overlap_counts['exact_window_pairs' if identical else 'partial_window_pairs'] += 1
                dest.write(json.dumps(dict(source=list(source), left=left, right=right,
                                           overlapping_frames=count, cross_split=cross, exact_window=identical))+'\n')
    reference_leaks = []
    for source,refs in reference_uses.items():
        for ref in refs:
            for target in windows.get(source,[]):
                if ref['split'] != target['split'] and integer(ref['reference_frame']) and target['start'] <= ref['reference_frame'] < target['end']:
                    reference_leaks.append(dict(source=list(source), reference=ref, target=target))
    if overlap_counts['cross_split']:
        finding('cross_split_source_window_overlap', overlap_counts['cross_split'])
    if reference_leaks:
        finding('cross_split_reference_in_target', reference_leaks)
    for group in content_candidates:
        if group['cross_split']:
            finding('cross_split_target_media_crc_size_candidate', group, 'warning')
    if declared.get('unique_target_windows') != len(exact) or declared.get('reused_target_windows') != sum(len(v)-1 for v in exact.values()):
        finding('declared_target_window_counts_mismatch', dict(actual_unique=len(exact), actual_reused=sum(len(v)-1 for v in exact.values()),
                                                             declared_unique=declared.get('unique_target_windows'), declared_reused=declared.get('reused_target_windows')))
    after = archive.stat()
    unchanged = (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    if not unchanged:
        finding('archive_stat_changed_during_audit', dict(before=[before.st_size,before.st_mtime_ns],after=[after.st_size,after.st_mtime_ns]))
    has_errors = any(i['severity']=='error' for i in global_issues) or any(k.startswith('error:') for k in issue_counts)
    has_warnings = any(i['severity']=='warning' for i in global_issues) or any(k.startswith('warning:') for k in issue_counts)
    summary = dict(status='METADATA_FINDINGS' if has_errors else ('METADATA_WARNINGS' if has_warnings else 'METADATA_CHECKS_COMPLETED'),
                   started_utc=stamp, finished_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=round(time.monotonic()-started,2), archive=str(archive),
                   archive_bytes=before.st_size, archive_mtime_ns=before.st_mtime_ns,
                   archive_size_and_mtime_unchanged=unchanged, archive_open_mode='r',
                   expected_records=expected_records, records=len(rows), completed_records=sum(r['audit_completed'] for r in results),
                   evidence_reuse=dict(enabled=recheck_existing, cached_records=len(rows)-len(fresh_rows),
                                       freshly_audited_records=len(fresh_rows),
                                       freshly_read_metadata_bytes=sum(r['metadata_uncompressed_bytes'] for r in results
                                                                       if not recheck_existing or r['record_index'] in recheck_ids),
                                       previous_full_audit_finished_utc=previous_summary['finished_utc'] if previous_summary else None,
                                       previous_full_audit_elapsed_seconds=previous_summary['elapsed_seconds'] if previous_summary else None),
                   split_counts=split_counts, task_counts=dict(Counter(r['task'] for r in rows)),
                   archive_entries=len(infos), unique_archive_paths=len(members),
                   archived_sample_json_count=len(archived_samples), episodes=len(groups['episode_id']),
                   metadata_json_read_count=json_count, metadata_uncompressed_bytes_read=json_bytes,
                   per_record_dependency_checks=dependency_count,
                   metadata_error_records=sum(any(i['severity']=='error' for i in r['issues']) for r in results),
                   metadata_warning_records=sum(any(i['severity']=='warning' for i in r['issues']) for r in results),
                   issue_counts={k:dict(occurrences=v, records=len(issue_records[k])) for k,v in sorted(issue_counts.items())},
                   metrics=dict(metric_totals), unique_target_windows=len(exact),
                   clock_rates={k:v for k,v in rate_report.items() if k in ('tolerance_fraction','target_windows','source_segments','target_windows_flagged','source_segments_flagged','records_in_flagged_source_segments','by_split','by_episode')},
                   reused_target_window_records=sum(len(v)-1 for v in exact.values()),
                   repeated_target_window_groups=len(duplicate_groups), source_window_overlap_pairs=dict(overlap_counts),
                   target_media_crc_size_candidate_groups=len(content_candidates),
                   cross_split_target_media_crc_size_candidates=sum(g['cross_split'] for g in content_candidates),
                   cross_split_reference_target_hits=len(reference_leaks), training_rounds=rounds,
                   global_issue_count=len(global_issues), global_issue_counts=dict(Counter(g['code'] for g in global_issues)),
                   scope=dict(metadata_only=True, visual_alignment_verified=False, media_decoded=False,
                              all_archive_media_crc_rechecked=False, paid_api_calls=0,
                              packaged_code_executed=False, external_source_files_opened=False),
                   limitations=[
                       'Declared CFR is 120 frames at 24 fps; metadata order does not prove encoded video frame count, PTS or correspondence.',
                       'Frame times are published estimates; monotonicity and internal agreement do not validate capture clock calibration.',
                       'Scene samples lack packaged camera/world tables: only their frame-state metadata can be checked here.',
                       'CRC32 plus size identifies duplicate candidates, not cryptographic byte identity or perceptual duplicates.',
                       'Within-split source-window reuse is reported separately from cross-split leakage and may represent different tasks.',
                       'Episode isolation does not establish unseen locations, visual dissimilarity or absence of semantically repeated scenes.',
                       'Reference chronology/source metadata does not prove the actual image was extracted from that frame.',
                       'Archived acceptance and visual review statements are not adopted as this audit evidence.'])
    dump_json(output/'summary.json',summary)
    dump_json(output/'global_issues.json',global_issues)
    dump_json(output/'duplicate_windows.json',duplicate_groups)
    dump_json(output/'media_duplicate_candidates.json',content_candidates)
    dump_json(output/'cross_split_reference_hits.json',reference_leaks)
    lines = ['# Delivery Metadata Reaudit', '', 'Metadata only. Visual alignment was NOT verified.', '',
             f"Records: {len(rows)}; completed: {summary['completed_records']}; errors: {summary['metadata_error_records']}; warnings: {summary['metadata_warning_records']}.",
             f"Splits: {split_counts}. Read {json_count} sample metadata JSON files ({json_bytes:,} uncompressed bytes).", '',
             '## Findings', '', '| Check | Records | Occurrences |', '|---|---:|---:|']
    for key,values in summary['issue_counts'].items():
        lines.append(f"| {key} | {values['records']} | {values['occurrences']} |")
    lines += ['', 'Global issues: '+json.dumps(summary['global_issue_counts']), '',
              f"Unique source windows: {len(exact)}; reused records: {summary['reused_target_window_records']}; cross-split overlapping window pairs: {overlap_counts['cross_split']}.",
              '', '## Example Records', '']
    shown = set()
    for result in results:
        for issue in result['issues']:
            if issue['code'] not in shown:
                shown.add(issue['code'])
                lines.append(f"- `{result['sample_id']}` ({result['split']}): {issue['code']}: `{json.dumps(issue['examples'][0])}`")
    lines += ['', '## Server / RGB PTS Rate', '',
              f"Target windows outside 2 percent: {rate_report['target_windows_flagged']}; source segments outside 2 percent: {len(flagged_sources)}; records in those sources: {len(affected_by_source_rate)}.",
              'Full elapsed ratios, fitted slopes and original endpoints: clock_rate_summary.json and source_clock_rates.jsonl. No timestamp corrections.', '']
    lines += ['', '## Scope Limits', ''] + ['- '+v for v in summary['limitations']]
    lines += ['', 'Full results: records.jsonl; anomalies.jsonl; frame_mappings.jsonl; source_window_overlaps.jsonl; duplicate_windows.json; global_issues.json.', '']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary, indent=2),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--expected-records',type=int,default=7085)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--recheck-existing',action='store_true',
                        help='Reuse completed same-archive record checks; reread error records and rebuild every summary')
    args = parser.parse_args()
    if args.self_test:
        success = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)).wasSuccessful()
        raise SystemExit(0 if success else 1)
    if args.workers < 1 or args.expected_records < 1:
        parser.error('workers and expected-records must be positive')
    archive, output = args.archive.resolve(), args.output.resolve()
    if archive == output or output in archive.parents:
        parser.error('Output must not contain the source archive')
    result = run(archive,output,args.workers,args.expected_records,args.recheck_existing)
    result = finalize_cross_checks(output)
    print(json.dumps(dict(final_status=result['status'], additional_cross_checks=result['additional_cross_checks'])),flush=True)
    raise SystemExit(2 if result['status']=='METADATA_FINDINGS' else 0)


if __name__ == '__main__':
    main()
