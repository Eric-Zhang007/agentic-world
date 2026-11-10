"""Streaming image-motion evidence. This is not 3D motion or contact detection."""
from pathlib import Path

import av
import cv2
import numpy as np


def track_points(previous, current, points, max_roundtrip_px=1.0):
    """Track supplied pixel anchors; invalid identities remain in output order.

    Round-trip consistency is a tracking diagnostic, not semantic identity proof.
    """
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError('Tracking requires equal-sized grayscale frames')
    if not np.isfinite(max_roundtrip_px) or max_roundtrip_px <= 0:
        raise ValueError('Roundtrip tolerance must be positive and finite')
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if not np.isfinite(source).all():
        raise ValueError('Nonfinite tracking anchors')
    n = len(source)
    result = dict(status='TRACKING_FAILED', points=np.full((n, 2), np.nan), valid=np.zeros(n, dtype=bool),
                  roundtrip_px=np.full(n, np.inf))
    if not n:
        return result
    moved, valid, _ = cv2.calcOpticalFlowPyrLK(previous, current, source, None)
    if moved is None:
        return result
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None)
    if back is None:
        return result
    height, width = previous.shape
    def inside(p):
        p = p.reshape(-1, 2)
        return (np.isfinite(p).all(axis=1) & (p[:,0] >= 0) & (p[:,0] < width)
                & (p[:,1] >= 0) & (p[:,1] < height))
    error = np.linalg.norm(source-back, axis=2).ravel()
    keep = ((valid.ravel() == 1) & (reverse.ravel() == 1)
            & inside(source) & inside(moved) & (error < max_roundtrip_px))
    result['points'][keep] = moved.reshape(-1, 2)[keep]
    result.update(status='TRACKED', valid=keep, roundtrip_px=error)
    return result


def pair_motion(previous, current, mask=None):
    points = cv2.goodFeaturesToTrack(previous, 400, .01, 8, mask=mask)
    if points is None or len(points) < 8:
        return {'status': 'INSUFFICIENT_FEATURES', 'tracks': 0}
    tracked = track_points(previous, current, points)
    if tracked['status'] == 'TRACKING_FAILED':
        return {'status': 'TRACKING_FAILED', 'tracks': 0}
    keep = tracked['valid']
    source, target = points.reshape(-1, 2)[keep], tracked['points'][keep]
    if mask is not None:
        pixels = np.rint(target).astype(int)
        inside = ((pixels[:,0] >= 0) & (pixels[:,0] < mask.shape[1]) &
                  (pixels[:,1] >= 0) & (pixels[:,1] < mask.shape[0]))
        allowed = np.zeros(len(target), dtype=bool)
        allowed[inside] = mask[pixels[inside,1], pixels[inside,0]] > 0
        source, target = source[allowed], target[allowed]
    if len(source) < 8:
        return {'status': 'INSUFFICIENT_TRACKS', 'tracks': len(source)}
    transform, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                   ransacReprojThreshold=2)
    if transform is None:
        return {'status': 'FIT_FAILED', 'tracks': len(source)}
    predicted = source @ transform[:, :2].T + transform[:, 2]
    displacement = np.linalg.norm(target-source, axis=1)
    residual = np.linalg.norm(target-predicted, axis=1)
    return {'status': 'MEASURED', 'tracks': len(source),
            'dominant_fit_fraction': float(inliers.mean()),
            'median_displacement_px': float(np.median(displacement)),
            'residual_p90_px': float(np.percentile(residual, 90)),
            'dominant_affine': transform.tolist()}


def inspect_motion(video, output, start=0., end=None, sample_interval=.1, width=640):
    if start < 0 or (end is not None and end <= start) or sample_interval <= 0:
        raise ValueError('Invalid inspection interval')
    if width < 32:
        raise ValueError('Analysis width must be at least 32 pixels')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, frames = [], []
    previous = None
    previous_time = None
    origin = None
    next_sample = start
    last = None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        for index, frame in enumerate(container.decode(stream)):
            pts = float(frame.time) if frame.time is not None else index/fps
            if origin is None:
                origin = pts
            timestamp = pts-origin
            if timestamp < start:
                continue
            if end is not None and timestamp > end + 1e-8:
                break
            image = frame.to_ndarray(format='rgb24')
            height = round(image.shape[0]*width/image.shape[1])
            resized = cv2.resize(image, (width, height))
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            if previous is not None:
                if timestamp <= previous_time:
                    raise ValueError('Non-increasing video timestamps')
                measured = pair_motion(previous, gray)
                rows.append(dict(measured, frame_id=index, time_sec=timestamp,
                                 delta_sec=timestamp-previous_time))
            previous, previous_time = gray, timestamp
            last = (index, timestamp, resized)
            if timestamp + 1e-8 >= next_sample:
                path = output/f'{index:06d}.jpg'
                cv2.imwrite(str(path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
                frames.append({'frame_id': index, 'timestamp': timestamp, 'path': str(path.resolve())})
                next_sample = timestamp + sample_interval
    if last is None:
        raise ValueError('No frames in requested interval')
    if not frames or frames[-1]['frame_id'] != last[0]:
        path = output/f'{last[0]:06d}.jpg'
        cv2.imwrite(str(path), cv2.cvtColor(last[2], cv2.COLOR_RGB2BGR))
        frames.append({'frame_id': last[0], 'timestamp': last[1], 'path': str(path.resolve())})
    measured = [r for r in rows if r['status'] == 'MEASURED']
    peaks = sorted(measured, key=lambda r: r['residual_p90_px']/r['delta_sec'], reverse=True)[:8]
    return {'video': str(Path(video).resolve()), 'start_sec': start, 'end_sec': last[1],
            'analysis_width': width, 'pair_count': len(rows), 'measured_pairs': len(measured),
            'frames': frames, 'motion_peaks': peaks, 'pairs': rows,
            'limitations': [
                'Image pixels at analysis_width, not world displacement or depth.',
                'Dominant affine motion may be camera motion OR a dominant foreground object.',
                'Residual motion may be articulation, occlusion, lighting or rendering artifacts.',
                'No object identity, contact, damage, ownership or completion is inferred.',
                'Sparse visible samples cannot prove events in unseen intervals.']}
