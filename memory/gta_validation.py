"""Conservative data eligibility checks, independent of appearance inference."""
import math


def camera_exclusion(frame):
    if frame.get('status') != 'estimated_alignment':
        return frame.get('status') or 'missing_status'
    data = frame.get('camera_interpolation') or {}
    if not {'lower_client_sample_seq', 'upper_client_sample_seq', 'alpha'}.issubset(data):
        return 'unsupported_camera_interpolation'
    alpha = data['alpha']
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        return 'invalid_camera_alpha'
    return None


def validate_reference_identity(observations, oid):
    entities = []
    for obs in observations:
        matches = [e for e in obs['interpolated_entities'] if str(e['dataset_entity_id']) == str(oid)]
        if len(matches) != 1:
            raise ValueError('Reference identity missing or ambiguous')
        entities.append(matches[0])
    if len({e['model_hash'] for e in entities}) != 1:
        raise ValueError('Model changed; split reference revision before packaging')
    frames = [o['frame_metadata'] for o in observations]
    if len({f['server_clock_domain'] for f in frames}) != 1:
        raise ValueError('Reference spans different clocks')
    times = [f['server_time_estimate_ns'] for f in frames]
    if any(a >= b for a,b in zip(times,times[1:])):
        raise ValueError('References and target frames must be strictly chronological')
    return entities
