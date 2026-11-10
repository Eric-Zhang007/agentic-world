"""Actual ZIP image/array decode, ordered disk reads, local resumable evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip')
OUTPUT = ROOT / 'outputs/dataset_reaudit/images'
VERSION = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def member(root, value):
    if not isinstance(value, str):
        return None
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value:
        raise ValueError('Non-relative dependency: ' + value)
    return str(PurePosixPath(root) / p)


def plan_sample(row, sample, names):
    root = row['sample_root']
    uses, pairs, masks, crops = [], [], [], []
    def add(value, role):
        path = member(root, value)
        if path and PurePosixPath(path).suffix.lower() in ('.png', '.jpg', '.jpeg', '.npy', '.npz'):
            uses.append({'member': path, 'role': role})
            return path
        return None
    conditions = sample['conditions']
    references = conditions.get('reference_pairs') or [conditions.get('reference_pair')]
    for i, pair in enumerate(references):
        if not isinstance(pair, dict):
            continue
        rgb = add(pair.get('rgb'), f'reference[{i}].rgb')
        wb = add(pair.get('whitebox'), f'reference[{i}].whitebox')
        if rgb and wb:
            pairs.append({'role': f'reference[{i}]', 'rgb': rgb, 'whitebox': wb})
    target = {}
    for kind, value in [('rgb', sample['supervision'].get('target_rgb')),
                        ('whitebox', conditions.get('target_whitebox'))]:
        path = add(value, f'target.{kind}')
        if not path and value:
            # Existing video-sidecar PNG is supplemental; its absence is not a schema failure.
            candidate = str(PurePosixPath(value).with_suffix('.png'))
            if member(root, candidate) in names:
                path = add(candidate, f'target.{kind}.existing_png')
        target[kind] = path
    if all(target.values()):
        pairs.append({'role': 'target', **target})
    valid = add(sample['supervision'].get('valid_pixels'), 'valid_pixels')
    if valid:
        masks.append({'mask': valid, 'image': target['rgb'], 'role': 'valid_pixels'})
    for i, edge in enumerate(sample.get('object_references', [])):
        rgb = add(edge.get('reference_rgb'), f'object[{i}].rgb')
        wb = add(edge.get('reference_whitebox'), f'object[{i}].whitebox')
        mask = add(edge.get('reference_mask'), f'object[{i}].mask')
        crop = add(edge.get('reference_crop'), f'object[{i}].crop')
        add(edge.get('embedding'), f'object[{i}].embedding')
        if rgb and wb:
            pairs.append({'role': f'object[{i}]', 'rgb': rgb, 'whitebox': wb})
        if mask:
            masks.append({'mask': mask, 'image': rgb, 'role': f'object[{i}].mask'})
        if crop:
            crops.append({'crop': crop, 'bbox': edge.get('reference_bbox_xyxy'), 'image': rgb})
    # Include actual sample PNG/NumPy assets even when not training dependencies.
    declared = {u['member'] for u in uses}
    for path in names:
        if path.startswith(root + '/') and path.lower().endswith(('.png', '.npy', '.npz')) and path not in declared:
            uses.append({'member': path, 'role': 'supplemental_unreferenced'})
    return {'sample_id': row['sample_id'], 'sample_root': root, 'split': row['split'],
            'uses': uses, 'pairs': pairs, 'masks': masks, 'crops': crops,
            'target_png_pair_present': all(target.values()), 'reference_pair_count': len(references)}


def decode(z, info, roles):
    result = {'member': info.filename, 'header_offset': info.header_offset, 'zip_crc': info.CRC,
              'uncompressed_bytes': info.file_size, 'roles': sorted(roles), 'errors': [], 'diagnostics': []}
    try:
        data = z.read(info)  # Full member decompression and ZIP CRC verification.
        if info.filename.lower().endswith(('.npy', '.npz')):
            value = np.load(io.BytesIO(data), allow_pickle=False)
            arrays = dict(value) if isinstance(value, np.lib.npyio.NpzFile) else {'array': value}
            if isinstance(value, np.lib.npyio.NpzFile):
                value.close()
            result['arrays'] = {}
            for key, array in arrays.items():
                numeric = array.dtype.kind in 'biufc'
                finite = bool(np.isfinite(array).all()) if numeric else False
                result['arrays'][key] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                         'size': int(array.size), 'finite': finite,
                                         'norm': float(np.linalg.norm(array.astype(np.complex128))) if finite else None}
                if not numeric or not finite or not array.size:
                    result['errors'].append(f'invalid_embedding_array:{key}')
                if array.ndim != 1:
                    result['diagnostics'].append(f'embedding_nonvector_shape:{key}')
        else:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                array = np.asarray(im)
                result.update(format=im.format, mode=im.mode, shape=list(array.shape), dtype=str(array.dtype),
                              size_wh=list(im.size), frames=getattr(im, 'n_frames', 1))
            is_mask = any('mask' in role or role == 'valid_pixels' for role in roles) or 'mask' in PurePosixPath(info.filename).name
            if is_mask:
                nonzero = int(np.count_nonzero(array))
                result['mask'] = {'nonzero_elements': nonzero, 'total_elements': int(array.size),
                                  'nonzero_fraction': nonzero / array.size,
                                  'min': int(array.min()), 'max': int(array.max())}
                if array.dtype == np.uint8:
                    result['mask']['values'] = np.flatnonzero(np.bincount(array.ravel(), minlength=256)).tolist()
                if not nonzero:
                    result['diagnostics'].append('empty_mask_not_automatic_sample_rejection')
                if array.ndim != 2:
                    result['diagnostics'].append('mask_not_single_channel')
            elif any(role.endswith('.rgb') or role.endswith('.whitebox') or role.endswith('.existing_png') for role in roles):
                if array.ndim != 3 or array.shape[2] not in (3, 4):
                    result['diagnostics'].append('paired_image_unexpected_channels')
        result['readable'] = True
    except Exception as exc:
        result['readable'] = False
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result


def summarize(db, plans, output, total_members, archive):
    results = {p: json.loads(v) for p, v in db.execute('select member,result from results')}
    media_rows = {}
    media_dir = output.parent / 'media'
    media_identity = media_dir / 'run.json'
    media_ledger = media_dir / 'samples.jsonl'
    if media_identity.exists() and media_ledger.exists():
        identity = json.loads(media_identity.read_text(encoding='utf-8'))
        stat = archive.stat()
        if (Path(identity['archive']).resolve() == archive.resolve()
                and identity['size'] == stat.st_size and identity['mtime_ns'] == stat.st_mtime_ns):
            with media_ledger.open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # The independent writer may have an unfinished last line.
                    media_rows[item['sample_id']] = item
    issues, sample_rows = [], []
    coverage = Counter()
    for plan in plans:
        errors, diagnostics = [], []
        joins_pending = 0
        paths = sorted({u['member'] for u in plan['uses']})
        pending = [p for p in paths if p not in results]
        for p in paths:
            r = results.get(p, {})
            errors.extend({'member': p, 'code': e} for e in r.get('errors', []))
            diagnostics.extend({'member': p, 'code': e} for e in r.get('diagnostics', []))
        def dimensions(path):
            return results.get(path, {}).get('size_wh')
        for pair in plan['pairs']:
            a, b = dimensions(pair['rgb']), dimensions(pair['whitebox'])
            if a and b:
                coverage['rgb_whitebox_pairs_checked'] += 1
            if a and b and a != b:
                errors.append({'code': 'rgb_whitebox_size_mismatch', 'pair': pair, 'dimensions': [a, b]})
        for pair in plan['masks']:
            a, b = dimensions(pair['mask']), dimensions(pair['image'])
            origin = 'png'
            if not pair['image']:
                video = media_rows.get(plan['sample_id'], {}).get('media', {}).get('target_rgb', {})
                if video.get('complete') and len(video.get('dimensions', [])) == 1:
                    b = video['dimensions'][0]
                    origin = 'independent_video_decode'
            if a and b:
                coverage['mask_image_pairs_checked'] += 1
                coverage['mask_dimensions_from_' + origin] += 1
            if a and b and a != b:
                errors.append({'code': 'mask_image_size_mismatch', 'pair': pair, 'dimensions': [a, b], 'image_evidence': origin})
            elif not pair['image'] and not b:
                joins_pending += 1
                diagnostics.append({'code': 'valid_mask_target_png_unavailable_dimension_join_deferred_to_media', 'member': pair['mask']})
        for crop in plan['crops']:
            size, box = dimensions(crop['crop']), crop['bbox']
            if size and isinstance(box, list) and len(box) == 4:
                expected = [box[2] - box[0], box[3] - box[1]]
                if size != expected:
                    diagnostics.append({'code': 'crop_bbox_size_difference', 'member': crop['crop'], 'actual': size, 'bbox_extent': expected})
        row = {'sample_id': plan['sample_id'], 'split': plan['split'], 'members': len(paths),
               'complete': not pending, 'pending_members': len(pending), 'errors': errors, 'diagnostics': diagnostics,
               'dimension_joins_pending': joins_pending,
               'status': 'FAIL' if errors else ('PENDING' if pending or joins_pending else ('REVIEW_DIAGNOSTIC' if diagnostics else 'PASS')),
               'target_png_pair_present': plan['target_png_pair_present'], 'reference_pair_count': plan['reference_pair_count']}
        sample_rows.append(row)
        if errors or diagnostics:
            issues.append(row)
    for name, rows in [('samples.jsonl', sample_rows), ('anomalies.jsonl', issues), ('members.jsonl', results.values())]:
        with (output / name).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    array_shapes = Counter(str(a['shape']) + ':' + a['dtype'] for r in results.values() for a in r.get('arrays', {}).values())
    summary = {'updated_at': now(), 'archive': str(archive), 'samples_total': len(plans),
               'samples_complete': sum(r['complete'] for r in sample_rows), 'unique_members_total': total_members,
               'unique_members_read': len(results), 'unreadable_members': sum(not r['readable'] for r in results.values()),
               'samples_with_errors': sum(bool(r['errors']) for r in sample_rows),
               'samples_with_diagnostics': sum(bool(r['diagnostics']) for r in sample_rows),
               'embedding_shapes_dtypes': dict(array_shapes),
               'embedding_arrays': sum(array_shapes.values()),
               'nonfinite_embedding_arrays': sum(not a['finite'] for r in results.values() for a in r.get('arrays', {}).values()),
               'image_shapes_dtypes': dict(Counter(str(r['shape'])+':'+r['dtype'] for r in results.values() if 'size_wh' in r)),
               'reference_pair_counts': dict(Counter(p['reference_pair_count'] for p in plans)),
               'target_png_pairs': sum(p['target_png_pair_present'] for p in plans),
               'dimension_check_coverage': dict(coverage),
               'independent_video_records_available': len(media_rows),
               'independent_video_ledger': str(media_ledger),
               'independent_video_complete_records': sum(all(r.get('media', {}).get(k, {}).get('complete', False) for k in ('target_rgb', 'target_whitebox')) for r in media_rows.values()),
               'independent_video_status_counts': dict(Counter(r.get('status', 'UNKNOWN') for r in media_rows.values())),
               'sample_status_counts': dict(Counter(r['status'] for r in sample_rows)),
               'dimension_joins_pending': sum(r['dimension_joins_pending'] for r in sample_rows),
               'error_counts': dict(Counter(e['code'] for r in sample_rows for e in r['errors'])),
               'diagnostic_counts': dict(Counter(e['code'] for r in sample_rows for e in r['diagnostics'])),
               'empty_masks': sum(r.get('mask', {}).get('nonzero_elements') == 0 for r in results.values()),
               'mask_shapes_dtypes': dict(Counter(str(r.get('shape'))+':'+str(r.get('dtype')) for r in results.values() if 'mask' in r)),
               'complete': len(results) == total_members and all(r['complete'] and not r['dimension_joins_pending'] for r in sample_rows),
               'scope': 'Actual ZIP CRC/decode and dimensions only. Empty masks are diagnostics, not invalid samples. No semantic, temporal or visual alignment pass.'}
    dump(output / 'summary.json', summary)
    (output / 'REPORT.md').write_text('# Image, Mask and Embedding Audit\n\n' +
        f"Samples complete: {summary['samples_complete']}/{len(plans)}. Unique members read: {len(results)}/{total_members}.\n\n" +
        f"Unreadable members: {summary['unreadable_members']}. Samples with structural errors: {summary['samples_with_errors']}. Empty masks: {summary['empty_masks']} (diagnostic only).\n\n" +
        f"Embedding arrays: {summary['embedding_arrays']}; nonfinite: {summary['nonfinite_embedding_arrays']}; shapes/dtypes: {dict(array_shapes)}.\n\n" +
        f"Dimension checks: {dict(coverage)}. Matching-archive independent video records available: {len(media_rows)}.\n\n" +
        f"Image audit sample statuses: {summary['sample_status_counts']}. Pending dimension joins: {summary['dimension_joins_pending']}.\n\n" +
        f"Independent video statuses (not image audit verdicts): {summary['independent_video_status_counts']}. Completed RGB/whitebox decode records: {summary['independent_video_complete_records']}/{len(plans)}.\n\n" +
        f"Diagnostic counts: {summary['diagnostic_counts']}.\n\n" +
        'Reference paths follow sample declarations, including multiple reference pairs. Existing target PNG sidecars and other sample PNG/NumPy assets are included.\n\n' +
        'See samples.jsonl for per-sample joins, members.jsonl for actual decode evidence, anomalies.jsonl for errors and diagnostics, and summary.json for distributions.\n\n' +
        'results.sqlite is the resume checkpoint. ZIP members are deduplicated by exact member path and read in header_offset order. No package files were modified.\n\n' + summary['scope'] + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit-members', type=int, help='Limit new reads for pilot; rerun without it to resume')
    parser.add_argument('--summarize-only', action='store_true', help='Refresh reports from checkpoints and independent video evidence without reading ZIP members')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stat = args.archive.stat()
    fingerprint = {'version': VERSION, 'archive': str(args.archive.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    db = sqlite3.connect(args.output / 'results.sqlite')
    db.execute('create table if not exists meta (key text primary key,value text)')
    db.execute('create table if not exists plans (root text primary key,value text)')
    db.execute('create table if not exists results (member text primary key,result text)')
    previous = db.execute("select value from meta where key='fingerprint'").fetchone()
    if previous and json.loads(previous[0]) != fingerprint:
        raise ValueError('Archive or audit version changed; use a fresh output directory')
    if args.summarize_only:
        plans = [json.loads(v) for v, in db.execute('select value from plans')]
        total = len({u['member'] for p in plans for u in p['uses']})
        summary = summarize(db, plans, args.output, total, args.archive)
        status = {'at': summary['updated_at'], 'stage': 'complete' if summary['complete'] else 'incomplete',
                  **{k: v for k, v in summary.items() if k not in ('updated_at', 'image_shapes_dtypes')}}
        dump(args.output / 'status.json', status)
        with (args.output / 'progress.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(status, allow_nan=False) + '\n')
        print(json.dumps(status), flush=True)
        db.close()
        return
    db.execute('insert or replace into meta values (?,?)', ('fingerprint', json.dumps(fingerprint)))
    db.commit()
    started = time.monotonic()
    with zipfile.ZipFile(args.archive) as z, (args.output / 'progress.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        def progress(**values):
            row = {'at': now(), 'elapsed_seconds': round(time.monotonic() - started, 2), **values}
            log.write(json.dumps(row) + '\n')
            dump(args.output / 'status.json', row)
            print(json.dumps(row), flush=True)
        infos = {i.filename: i for i in z.infolist()}
        records = [json.loads(line) for split in ('train', 'validation', 'test')
                   for line in z.read(f'manifests/{split}.jsonl').splitlines()]
        roots = {r['sample_root']: r for r in records}
        sample_names = {root: [] for root in roots}
        for name in infos:
            parts = name.split('/')
            if len(parts) >= 3 and '/'.join(parts[:2]) in sample_names:
                sample_names['/'.join(parts[:2])].append(name)
        cached = {r for r, in db.execute('select root from plans')}
        ordered = sorted(roots, key=lambda r: infos[r + '/sample.json'].header_offset)
        progress(stage='planning', samples_total=len(roots), samples_cached=len(cached))
        for index, root in enumerate(ordered, 1):
            if root not in cached:
                sample = json.loads(z.read(root + '/sample.json'))
                plan = plan_sample(roots[root], sample, sample_names[root])
                db.execute('insert into plans values (?,?)', (root, json.dumps(plan)))
            if index % 250 == 0:
                db.commit()
                progress(stage='planning', samples_planned=index, samples_total=len(roots))
        db.commit()
        plans = [json.loads(v) for v, in db.execute('select value from plans')]
        uses = {}
        for plan in plans:
            for use in plan['uses']:
                uses.setdefault(use['member'], set()).add(use['role'])
        done = {p for p, in db.execute('select member from results')}
        pending = sorted((p for p in uses if p not in done), key=lambda p: infos[p].header_offset if p in infos else -1)
        progress(stage='decoding', unique_members_total=len(uses), completed=len(done), remaining=len(pending))
        new = 0
        for path in pending:
            if args.limit_members is not None and new >= args.limit_members:
                break
            result = decode(z, infos[path], uses[path]) if path in infos else {
                'member': path, 'readable': False, 'errors': ['missing_member'], 'diagnostics': []}
            db.execute('insert into results values (?,?)', (path, json.dumps(result, allow_nan=False)))
            new += 1
            if new % 100 == 0:
                db.commit()
                progress(stage='decoding', completed=len(done)+new, unique_members_total=len(uses), last_member=path)
        db.commit()
        summary = summarize(db, plans, args.output, len(uses), args.archive)
        progress(stage='complete' if summary['complete'] else 'pilot_complete', **{k:v for k,v in summary.items() if k not in ('updated_at', 'archive', 'scope')})
    db.close()


if __name__ == '__main__':
    main()
