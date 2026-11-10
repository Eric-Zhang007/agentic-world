"""Decode delivered ZIP bytes completely. Structural audit only, no visual pass.

Run --limit 10 for a pilot, then omit --limit with the same output to resume.
One isolated worker holds the read-only ZIP; timeouts restart it. No extraction.
"""
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import io
import json
import multiprocessing as mp
import os
import subprocess
from pathlib import Path, PurePosixPath
import time
import zipfile
from PIL import Image, ImageDraw

import av

VERSION = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(40):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.1)


def decode(z, member):
    started = time.monotonic()
    result = dict(member=member, decoded_frames=0, errors=[], complete=False)
    times, sizes, thumbnails = [], set(), {}
    try:
        info = z.getinfo(member)
        result.update(archive_bytes=info.file_size, zip_crc=info.CRC)
        # ZipFile.read also verifies this member's CRC; only one video is resident.
        with av.open(io.BytesIO(z.read(member)), mode='r') as container:
            if len(container.streams.video) != 1:
                raise ValueError(f'Expected one video stream, got {len(container.streams.video)}')
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            result.update(codec=stream.codec_context.name, header_frames=stream.frames,
                          fps=float(stream.average_rate) if stream.average_rate else None,
                          fps_rational=str(stream.average_rate), time_base=str(stream.time_base),
                          header_size=[stream.width, stream.height])
            corrupt = 0
            for frame in container.decode(stream):
                index = result['decoded_frames']
                if index in (0, 30, 60, 90, 119):
                    thumbnails[index] = frame.to_image().resize((320, 180), Image.Resampling.BILINEAR)
                result['decoded_frames'] += 1
                sizes.add((frame.width, frame.height))
                times.append(float(frame.pts*frame.time_base) if frame.pts is not None and frame.time_base else None)
                corrupt += int(frame.is_corrupt)
            result['complete'] = True
            result['corrupt_frames'] = corrupt
            if corrupt:
                result['errors'].append('corrupt_frames')
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    missing = sum(t is None for t in times)
    deltas = [b-a for a, b in zip(times, times[1:]) if a is not None and b is not None]
    nonmonotonic = sum(d <= 0 for d in deltas)
    result.update(dimensions=[list(s) for s in sorted(sizes)], missing_timestamps=missing,
                  nonmonotonic_timestamps=nonmonotonic,
                  duplicate_timestamps=sum(d == 0 for d in deltas),
                  backwards_timestamps=sum(d < 0 for d in deltas),
                  timestamp_anomalies=[dict(previous_frame=i, frame=i+1, previous_time=a, time=b)
                                       for i, (a, b) in enumerate(zip(times, times[1:]))
                                       if a is not None and b is not None and b <= a][:30],
                  first_time=times[0] if times else None, last_time=times[-1] if times else None,
                  min_frame_interval=min(deltas) if deltas else None,
                  max_frame_interval=max(deltas) if deltas else None,
                  measured_fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 and not missing and times[-1]>times[0] else None,
                  seconds=time.monotonic()-started)
    for condition, message in ((not times, 'no_frames'), (missing, 'missing_timestamps'),
                               (nonmonotonic, 'nonmonotonic_timestamps'),
                               (len(sizes) != 1, 'nonconstant_or_missing_dimensions')):
        if condition:
            result['errors'].append(message)
    return result, times, thumbnails


def audit(z, entry):
    started = time.monotonic()
    result = dict(key=entry['key'], split=entry['split'], sample_id=entry.get('sample_id'),
                  sample_root=entry.get('sample_root'), errors=[], media={}, started_utc=now())
    root = entry.get('sample_root', '')
    sample = {}
    try:
        sample = json.loads(z.read(str(PurePosixPath(root)/'sample.json')))
        result['declared_frame_count'] = sample.get('frame_count')
        result['declared_fps'] = sample.get('fps')
    except Exception as exc:
        result['errors'].append(f'sample_json: {type(exc).__name__}: {exc}')
    timelines, pictures = {}, {}
    for name, section in (('target_rgb', 'supervision'), ('target_whitebox', 'conditions')):
        declared = sample.get(section, {}).get(name, name+'.mp4')
        member = str(PurePosixPath(root)/declared)
        media, timelines[name], pictures[name] = decode(z, member)
        result['media'][name] = media
        if media['decoded_frames'] != sample.get('frame_count'):
            media['errors'].append('declared_frame_count_mismatch')
        fps = sample.get('fps')
        for field in ('fps', 'measured_fps'):
            if not isinstance(fps, (int, float)) or fps <= 0 or media.get(field) is None or abs(media[field]-fps) > max(.001, fps*.001):
                media['errors'].append('declared_'+field+'_mismatch')
        if media['errors']:
            result['errors'].append(name+'_failed')
    a, b = result['media'].values()
    for field in ('decoded_frames', 'dimensions', 'fps'):
        if a.get(field) != b.get(field):
            result['errors'].append('rgb_proxy_'+field+'_mismatch')
    ta, tb = timelines.values()
    matching = len(ta) == len(tb) and all(x is not None and y is not None and abs(x-y) <= 1e-6 for x, y in zip(ta, tb))
    result['rgb_proxy_timestamps_match'] = matching
    if not matching:
        result['errors'].append('rgb_proxy_timestamps_mismatch')
    sheet = Image.new('RGB', (640, 900), '#202020')
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate((0, 30, 60, 90, 119)):
        for column, name in enumerate(('target_rgb', 'target_whitebox')):
            tile = pictures[name].get(index)
            if tile is not None:
                sheet.paste(tile, (column*320, row*180))
            label = f'{name} frame {index}' + (' MISSING' if tile is None else '')
            draw.text((column*320+5, row*180+5), label, fill='white', stroke_width=2, stroke_fill='black')
    buffer = io.BytesIO()
    sheet.save(buffer, format='JPEG', quality=85)
    result['_contact_jpeg'] = buffer.getvalue()
    result['contact_frames'] = {k: sorted(v) for k, v in pictures.items()}
    result.update(status='STRUCTURAL_FAIL' if result['errors'] else 'STRUCTURAL_PASS',
                  seconds=time.monotonic()-started, finished_utc=now())
    return result


def worker(archive, connection):
    with zipfile.ZipFile(archive, 'r') as z:
        while True:
            entry = connection.recv()
            if entry is None:
                break
            try:
                result = audit(z, entry)
            except Exception as exc:
                result = dict(key=entry['key'], split=entry['split'], sample_id=entry.get('sample_id'),
                              status='STRUCTURAL_FAIL', errors=[f'worker: {type(exc).__name__}: {exc}'])
            connection.send(result)


def restored(path):
    records = {}
    if path.exists():
        with path.open('r+b') as handle:
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    if not line.endswith(b'\n'):
                        raise ValueError('incomplete final record')
                    record = json.loads(line)
                    records[record['key']] = record
                except (ValueError, KeyError):
                    if handle.read():
                        raise ValueError('Corrupt non-final JSONL record; refusing unsafe resume')
                    handle.truncate(start)
                    break
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip'))
    parser.add_argument('--output', type=Path, default=Path('outputs/dataset_reaudit/media'))
    parser.add_argument('--limit', type=int, help='Total completed records to reach, including resumed records')
    parser.add_argument('--timeout', type=float, default=180, help='Seconds per sample before recording failure and continuing')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ffprobe-sample', help='Independent frame timestamp check from ZIP bytes; no audit worker changes')
    parser.add_argument('--benchmark-cache', type=Path, help='Benchmark 10 samples in a small SSD ZIP; no formal audit writes')
    parser.add_argument('--final-report', action='store_true', help='Refresh final classification after complete coverage')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be between 1 and 8')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.final_report:
        final_report(args)
        return
    if args.benchmark_cache:
        benchmark_cache(args)
        return
    if args.ffprobe_sample:
        ffprobe_check(args)
        return
    # OS lock is released even on process crash; prevents concurrent JSONL writers.
    import msvcrt
    with (args.output/'audit.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        run(args)


def final_report(args):
    """Classify latest records without decoding again or modifying the archive."""
    records = {}
    for line in (args.output/'samples.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        records[row['key']] = row
    entries = {}
    with zipfile.ZipFile(args.archive) as archive:
        for split in ('train', 'validation', 'test'):
            for index, line in enumerate(archive.read(f'manifests/{split}.jsonl').splitlines()):
                if line.strip():
                    entries[f'{split}:{index}'] = dict(json.loads(line), split=split)
    missing = set(entries)-set(records)
    if missing:
        raise ValueError(f'Cannot publish final report: {len(missing)} samples not audited')
    counts = Counter()
    groups = {field: {} for field in ('episode_id', 'task', 'split')}
    failures = []
    media_counts = Counter()
    sample_errors, media_errors = Counter(), Counter()
    contact_missing = []
    for key, entry in entries.items():
        row = records[key]
        sample_errors.update(row.get('errors', []))
        media = list(row.get('media', {}).values())
        duplicate = any(m.get('duplicate_timestamps', 0) for m in media)
        backwards = any(m.get('backwards_timestamps', 0) for m in media)
        mismatch = row.get('rgb_proxy_timestamps_match') is False
        decode_failure = len(media) != 2 or any(not m.get('complete') or not m.get('decoded_frames') or m.get('corrupt_frames', 0) for m in media)
        counts['samples'] += 1
        counts['duplicate_pts_samples'] += int(duplicate)
        counts['backwards_pts_samples'] += int(backwards)
        counts['rgb_proxy_time_mismatch_samples'] += int(mismatch)
        counts['decode_failure_samples'] += int(decode_failure)
        counts['duplicate_and_time_mismatch_samples'] += int(duplicate and mismatch)
        counts['time_mismatch_without_duplicate_samples'] += int(mismatch and not duplicate)
        counts['duplicate_without_time_mismatch_samples'] += int(duplicate and not mismatch)
        for m in media:
            media_errors.update(m.get('errors', []))
            media_counts['videos'] += 1
            media_counts['complete_decodes'] += int(m.get('complete', False))
            media_counts['decoded_frames'] += m.get('decoded_frames', 0)
            media_counts['duplicate_pts_videos'] += int(m.get('duplicate_timestamps', 0) > 0)
            media_counts['duplicate_pts_events'] += m.get('duplicate_timestamps', 0)
            media_counts['backwards_pts_videos'] += int(m.get('backwards_timestamps', 0) > 0)
            media_counts['declared_frame_count_mismatch_videos'] += int('declared_frame_count_mismatch' in m.get('errors', []))
            media_counts['missing_timestamp_videos'] += int(m.get('missing_timestamps', 0) > 0)
        failed = row['status'] != 'STRUCTURAL_PASS'
        classification = ('DECODE_FAIL' if decode_failure else
                          'DECODE_PASS_TIMING_FAIL' if failed and (duplicate or backwards or mismatch) else
                          'DECODE_PASS_OTHER_STRUCTURE_FAIL' if failed else 'STRUCTURAL_PASS')
        counts[classification] += 1
        contact = row.get('contact_sheet')
        if not contact or not (args.output/contact).is_file():
            contact_missing.append(key)
        for field, mapping in groups.items():
            value = str(entry.get(field, 'unknown'))
            stats = mapping.setdefault(value, Counter())
            stats['total_samples'] += 1
            stats['affected_samples'] += int(failed)
            stats['duplicate_pts_samples'] += int(duplicate)
            stats['rgb_proxy_time_mismatch_samples'] += int(mismatch)
            stats['decode_failure_samples'] += int(decode_failure)
        if failed:
            failures.append(dict(key=key, sample_id=entry['sample_id'], episode_id=entry.get('episode_id'),
                                 task=entry.get('task'), split=entry['split'], classification=classification,
                                 duplicate_pts=duplicate, backwards_pts=backwards, rgb_proxy_time_mismatch=mismatch,
                                 errors=row.get('errors', []), media=row.get('media', {}), contact_sheet=contact))
    ffprobe = []
    for path in sorted(args.output.glob('ffprobe_*.json')):
        evidence = json.loads(path.read_text())
        ffprobe.append(dict(sample_id=evidence['sample_id'], evidence_file=path.name,
                            classification=evidence['classification'],
                            media={k: {f:v.get(f) for f in ('decoded_frames', 'duplicate_pts', 'backwards_pts', 'anomalies')}
                                   for k,v in evidence['media'].items()}))
    report = dict(completed_utc=now(), scope='Structural integrity only, not visual acceptance; timing failure is not video corruption.',
                  archive=str(args.archive), complete_coverage=True, counts=dict(counts), media_counts=dict(media_counts),
                  affected_samples=len(failures), contact_sheets=len(entries)-len(contact_missing),
                  sample_error_counts=dict(sample_errors), media_error_counts=dict(media_errors),
                  missing_contact_keys=contact_missing, groups=groups, ffprobe_evidence=ffprobe,
                  grouping_note='Failure categories overlap; deduplicated by stable split:index, latest JSONL record wins.')
    save_json(args.output/'final_report.json', report)
    with (args.output/'failures_classified.jsonl').open('w', encoding='utf-8') as out:
        for row in failures:
            out.write(json.dumps(row, allow_nan=False)+'\n')
    lines = ['# Final delivery media audit', '', report['scope'], '',
             f"Coverage: {counts['samples']} samples, {media_counts['videos']} videos; affected samples: {len(failures)}.",
             f"Contact sheets: {report['contact_sheets']}; decode failures: {counts['decode_failure_samples']}.", '',
             '## Classification', '', '| Metric | Samples |', '| --- | ---: |']
    for field in ('duplicate_pts_samples', 'rgb_proxy_time_mismatch_samples', 'backwards_pts_samples', 'decode_failure_samples',
                  'duplicate_and_time_mismatch_samples', 'duplicate_without_time_mismatch_samples', 'time_mismatch_without_duplicate_samples'):
        lines.append(f'| {field} | {counts[field]} |')
    for field, mapping in groups.items():
        lines += ['', f'## {field}', '', '| Group | Total | Affected | Duplicate PTS | RGB/proxy time mismatch | Decode failure |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for value, stats in sorted(mapping.items()):
            if field == 'episode_id' and not stats['affected_samples']:
                continue
            lines.append('| '+value+' | '+' | '.join(str(stats[k]) for k in ('total_samples', 'affected_samples', 'duplicate_pts_samples', 'rgb_proxy_time_mismatch_samples', 'decode_failure_samples'))+' |')
    lines += ['', '## Independent ffprobe evidence', '']
    lines += [f"- {e['sample_id']}: {e['evidence_file']}" for e in ffprobe]
    (args.output/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(counts=counts, media_counts=media_counts, affected_samples=len(failures),
                         contact_sheets=report['contact_sheets'], ffprobe_samples=len(ffprobe)), indent=2))


def benchmark_cache(args):
    """Matched serial decode benchmark; running formal workers are untouched."""
    completed = set()
    ledger = args.output/'samples.jsonl'
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            try:
                completed.add(json.loads(line)['key'])
            except ValueError:
                pass
    cache = args.benchmark_cache
    cache.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    target = cache/f'media_benchmark_{stamp}.zip'
    report = dict(started_utc=now(), source=str(args.archive), cache=str(target),
                  formal_results=False, workers=1, codec_threads=1,
                  caveats=['Formal E-drive audit continues concurrently.',
                           'Source read during cache construction warms OS cache; no cache eviction.',
                           'Small subset ZIP central directory is cheaper; open time reported separately.',
                           'Benchmark includes full decode and in-memory contact sheet creation, not result/image disk writes.'])
    opened = time.monotonic()
    with zipfile.ZipFile(args.archive) as source:
        report['source_open_seconds'] = time.monotonic()-opened
        entries = []
        for split in ('train', 'validation', 'test'):
            for i, line in enumerate(source.read(f'manifests/{split}.jsonl').splitlines()):
                if not line.strip() or f'{split}:{i}' in completed:
                    continue
                entry = json.loads(line)
                entry.update(key=f'{split}:{i}', split=split)
                member = str(PurePosixPath(entry['sample_root'])/'target_rgb.mp4')
                entry['offset'] = source.getinfo(member).header_offset
                entries.append(entry)
        entries.sort(key=lambda e: e['offset'])
        # A contiguous pending block away from the formal workers' current position.
        selected = entries[len(entries)//2:len(entries)//2+10]
        if len(selected) != 10:
            raise ValueError('Need at least 10 pending samples')
        report['samples'] = [dict(key=e['key'], sample_id=e['sample_id'], offset=e['offset']) for e in selected]
        copied = read_seconds = 0
        copy_started = time.monotonic()
        with zipfile.ZipFile(target, 'x') as dest:
            for entry in selected:
                root = PurePosixPath(entry['sample_root'])
                for name in ('sample.json', 'target_rgb.mp4', 'target_whitebox.mp4'):
                    member = str(root/name)
                    begin = time.monotonic()
                    payload = source.read(member)
                    read_seconds += time.monotonic()-begin
                    copied += len(payload)
                    # writestr mutates ZipInfo offsets; never share source metadata.
                    dest.writestr(copy.copy(source.getinfo(member)), payload)
        report.update(copy_seconds=time.monotonic()-copy_started, source_read_seconds=read_seconds,
                      uncompressed_bytes=copied, cache_zip_bytes=target.stat().st_size,
                      source_read_MB_per_second=copied/1e6/read_seconds)
        print(json.dumps({k:v for k,v in report.items() if k not in ('samples', 'caveats')}), flush=True)
        opened = time.monotonic()
        with zipfile.ZipFile(target) as ssd:
            report['cache_open_seconds'] = time.monotonic()-opened
            timings = {'E': [], 'D': []}
            signatures = {}
            # Alternate first drive to reduce ordering/temperature bias.
            for index, entry in enumerate(selected):
                for drive in (('E', 'D') if index % 2 == 0 else ('D', 'E')):
                    begin = time.monotonic()
                    result = audit(source if drive == 'E' else ssd, entry)
                    elapsed = time.monotonic()-begin
                    result.pop('_contact_jpeg', None)
                    if not all(m['complete'] and m['decoded_frames'] > 0 for m in result['media'].values()):
                        raise RuntimeError(f'Invalid benchmark: incomplete decode on {drive}: {result}')
                    signatures[(entry['key'], drive)] = [(m['decoded_frames'], m['dimensions'], m['errors']) for m in result['media'].values()]
                    timings[drive].append(elapsed)
                    print(json.dumps(dict(benchmark=True, sample=entry['sample_id'], drive=drive, seconds=elapsed)), flush=True)
            report['same_decode_results'] = all(signatures[(e['key'], 'E')] == signatures[(e['key'], 'D')] for e in selected)
            report['benchmark_valid'] = report['same_decode_results']
            report['timings'] = {k: dict(seconds=sum(v), samples_per_second=len(v)/sum(v), sample_seconds=v) for k,v in timings.items()}
            report['ssd_speedup'] = sum(timings['E'])/sum(timings['D'])
    report['finished_utc'] = now()
    save_json(args.output/'ssd_benchmark.json', report)
    print(json.dumps({k:report[k] for k in ('timings', 'ssd_speedup', 'same_decode_results')}, indent=2))


def ffprobe_check(args):
    """Independent ffprobe executable over seekable in-memory ZIP bytes."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    sid = args.ffprobe_sample
    if Path(sid).name != sid or '/' in sid or '\\' in sid:
        raise ValueError('Unsafe sample id')
    result = dict(sample_id=sid, tool='ffprobe', input='actual ZIP bytes via loopback HTTP byte ranges; no extracted media',
                  checked_utc=now(), scope='Decode and timing only; not visual acceptance', media={})
    with zipfile.ZipFile(args.archive) as z:
        for name in ('target_rgb', 'target_whitebox'):
            member = f'samples/{sid}/{name}.mp4'
            command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                       '-show_entries', 'frame=pts,pts_time,best_effort_timestamp_time,pkt_duration_time,width,height',
                       '-of', 'json']
            payload = z.read(member)
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *unused):
                    pass

                def do_GET(self):
                    start, end = 0, len(payload)-1
                    requested = self.headers.get('Range')
                    if requested:
                        first, last = requested.removeprefix('bytes=').split('-', 1)
                        start = int(first or 0)
                        end = min(int(last), end) if last else end
                    self.send_response(206 if requested else 200)
                    self.send_header('Content-Type', 'video/mp4')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Content-Length', str(end-start+1))
                    if requested:
                        self.send_header('Content-Range', f'bytes {start}-{end}/{len(payload)}')
                    self.end_headers()
                    try:
                        self.wfile.write(payload[start:end+1])
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
            with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    checked = subprocess.run(command+['-i', f'http://127.0.0.1:{server.server_port}/video.mp4'],
                                             capture_output=True, timeout=120)
                finally:
                    server.shutdown()
                    thread.join()
            if checked.returncode or checked.stderr:
                raise RuntimeError(checked.stderr.decode(errors='replace'))
            frames = json.loads(checked.stdout)['frames']
            pts = [float(f['pts_time']) for f in frames]
            anomalies = [dict(previous_frame=i, frame=i+1, previous_pts=frames[i]['pts'],
                              pts=frames[i+1]['pts'], previous_time=a, time=b, interval=b-a)
                         for i, (a, b) in enumerate(zip(pts, pts[1:])) if b-a <= 0 or b-a > 1.5/24]
            result['media'][name] = dict(member=member, decoded_frames=len(frames),
                                        duplicate_pts=sum(b==a for a,b in zip(pts,pts[1:])),
                                        backwards_pts=sum(b<a for a,b in zip(pts,pts[1:])),
                                        anomalies=anomalies, frames=frames)
    result['classification'] = 'DECODE_PASS_TIMING_FAIL' if any(m['duplicate_pts'] or m['backwards_pts'] for m in result['media'].values()) else 'DECODE_PASS'
    path = args.output/f'ffprobe_{sid}.json'
    save_json(path, result)
    print(json.dumps({**result, 'media': {k: {a:b for a,b in v.items() if a != 'frames'} for k,v in result['media'].items()}}, indent=2))


def run(args):
    stat = args.archive.stat()
    identity = dict(archive=str(args.archive.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    audit_version=VERSION, pyav=av.__version__, scope='Structural integrity only; not visual acceptance')
    identity_path = args.output/'run.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Archive or audit version changed; use a new output directory')
    save_json(identity_path, identity)
    groups = {}
    with zipfile.ZipFile(args.archive) as z:
        for split in ('train', 'validation', 'test'):
            entries = []
            for i, line in enumerate(z.read(f'manifests/{split}.jsonl').splitlines()):
                if line.strip():
                    entry = json.loads(line)
                    entry.update(split=split, key=f'{split}:{i}')
                    member = str(PurePosixPath(entry['sample_root'])/'target_rgb.mp4')
                    try:
                        entry['rgb_header_offset'] = z.getinfo(member).header_offset
                    except KeyError:
                        entry['rgb_header_offset'] = stat.st_size
                    entries.append(entry)
            groups[split] = entries
    # Physical ZIP order minimizes seeks on rotating media; identity stays split:index.
    entries = sorted((e for rows in groups.values() for e in rows),
                     key=lambda e: (e['rgb_header_offset'], e['key']))
    ledger = args.output/'samples.jsonl'
    records = restored(ledger)
    started = time.monotonic()
    # Old structural records without contact sheets need one cache-building decode.
    records = {k: r for k, r in records.items() if r.get('contact_attempted') and
               (not r.get('contact_sheet') or (args.output/r['contact_sheet']).exists())}
    initial = len(records)
    slots = []
    (args.output/'contact').mkdir(exist_ok=True)

    def progress(state, active=None):
        elapsed = time.monotonic()-started
        completed = len(records)-initial
        rate = completed/elapsed if completed else None
        summary = dict(state=state, pid=os.getpid(), updated_utc=now(), total=len(entries),
                       completed=len(records), remaining=len(entries)-len(records), resumed=initial,
                       split_totals={k: len(v) for k, v in groups.items()},
                       statuses=dict(Counter(r['status'] for r in records.values())),
                       elapsed_seconds=elapsed, samples_per_second=rate,
                       eta_seconds=(len(entries)-len(records))/rate if rate else None,
                       active=active, workers=args.workers, codec_threads=1,
                       scheduling='target_rgb_zip_header_offset', commit_order='completion', scope=identity['scope'])
        try:
            save_json(args.output/'progress.json', summary)
        except PermissionError as exc:
            print(f'Progress refresh deferred; durable JSONL retained: {exc}', flush=True)
        return summary

    try:
        todo = [e for e in entries if e['key'] not in records]
        if args.limit:
            todo = todo[:max(0, args.limit-len(records))]
        dispatched = committed = 0
        pending = []
        last_progress = 0
        with ledger.open('a', encoding='utf-8') as out:
            while committed < len(todo):
                while len(slots) < args.workers:
                    parent, child = mp.Pipe()
                    process = mp.Process(target=worker, args=(str(args.archive), child))
                    process.start()
                    child.close()
                    slots.append(dict(process=process, connection=parent, entry=None))
                for slot in slots[:]:
                    if slot['entry'] is None and dispatched < len(todo):
                        slot.update(entry=todo[dispatched], index=dispatched, started=time.monotonic())
                        slot['connection'].send(slot['entry'])
                        dispatched += 1
                    if slot['entry'] is None:
                        continue
                    connection = slot['connection']
                    result = None
                    try:
                        if connection.poll():
                            result = connection.recv()
                        elif time.monotonic()-slot['started'] > args.timeout or not slot['process'].is_alive():
                            raise TimeoutError('sample timed out or worker exited')
                    except (EOFError, OSError, TimeoutError) as exc:
                        entry = slot['entry']
                        result = dict(key=entry['key'], split=entry['split'], sample_id=entry.get('sample_id'),
                                      status='STRUCTURAL_FAIL', errors=[str(exc)], finished_utc=now())
                        slot['process'].terminate()
                        slot['process'].join()
                        connection.close()
                        slots.remove(slot)
                    if result is not None:
                        result['rgb_header_offset'] = slot['entry']['rgb_header_offset']
                        pending.append(result)
                        slot['entry'] = None
                while pending:
                    result = pending.pop(0)
                    jpeg = result.pop('_contact_jpeg', None)
                    result['contact_attempted'] = True
                    if jpeg is not None:
                        sid = result['sample_id']
                        if not sid or Path(sid).name != sid or '/' in sid or '\\' in sid:
                            raise ValueError('Unsafe contact sheet sample id')
                        relative = f'contact/{sid}.jpg'
                        temporary = args.output/(relative+'.tmp')
                        temporary.write_bytes(jpeg)
                        temporary.replace(args.output/relative)
                        result['contact_sheet'] = relative
                    out.write(json.dumps(result, allow_nan=False)+'\n')
                    out.flush()
                    os.fsync(out.fileno())
                    records[result['key']] = result
                    committed += 1
                    summary = progress('running')
                    print(json.dumps({k: summary[k] for k in ('completed', 'total', 'statuses', 'eta_seconds')}), flush=True)
                if time.monotonic()-last_progress > 5:
                    progress('running', [s['entry'] for s in slots if s['entry'] is not None])
                    last_progress = time.monotonic()
                time.sleep(.05)
        summary = progress('complete' if len(records) == len(entries) else 'pilot_complete')
        save_json(args.output/('summary.json' if summary['state'] == 'complete' else 'pilot.json'), summary)
    finally:
        for slot in slots:
            slot['process'].terminate()
            slot['process'].join()
            slot['connection'].close()


if __name__ == '__main__':
    mp.freeze_support()
    main()
