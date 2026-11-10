"""Experimental temporal gating for the pinned native H3 control branch.

This is not a trained sparse-control mode. Pixel windows are expanded to H3's
temporal tokens; self-attention can still propagate the selected constraints.
"""
def frame_spans(frame_count):
    if frame_count < 5 or frame_count % 17 != 5:
        raise ValueError("H3 frames must be 17*n+5")
    spans, start = [], 0
    while start < frame_count:
        stop = start + (1, 4, 4, 4, 4)[len(spans) % 5]
        if stop > frame_count:
            raise ValueError("Invalid temporal token boundary")
        spans.append((start, stop))
        start = stop
    return spans


def temporal_schedule(frame_count, keyframes):
    spans = frame_spans(frame_count)
    if keyframes is None:
        active = [True] * len(spans)
    else:
        if not keyframes or any(type(k) is not int or not 0 <= k < frame_count for k in keyframes):
            raise ValueError("keyframes must contain valid integer frame indices")
        active = [any(start <= k < stop for k in keyframes) for start, stop in spans]
    frames = [index for enabled, (start, stop) in zip(active, spans) if enabled
              for index in range(start, stop)]
    return {"frame_count": frame_count, "requested_keyframes": keyframes,
            "latent_weights": [float(value) for value in active],
            "active_frame_indices": frames, "token_spans": spans,
            "active_fraction": len(frames) / frame_count}


def gate_tensors(schedule, latent, layout):
    import torch
    weights = torch.tensor(schedule["latent_weights"], device=latent.device, dtype=latent.dtype)
    if latent.shape[2] != len(weights):
        raise ValueError("Control latent and pixel-to-token mapping disagree")
    segments = [(a, b) for a, b, kind in layout.segments if kind == "video"]
    if len(segments) != 1:
        raise ValueError("Expected one target video segment")
    start, stop = segments[0]
    if (stop - start) % len(weights):
        raise ValueError("Video rows do not divide into temporal tokens")
    rows = torch.zeros(layout.seq_len, device=latent.device, dtype=latent.dtype)
    rows[start:stop] = weights.repeat_interleave((stop - start) // len(weights))
    return latent * weights[None, None, :, None, None], rows
