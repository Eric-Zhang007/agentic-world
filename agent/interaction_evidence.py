"""Optional temporal inspection using the existing tool-calling client."""
import json
from pathlib import Path

from agent.action_planner import image_content
from agent.client import AgentClient
from perception.motion_evidence import inspect_motion


INSPECT_TOOL = {'type': 'function', 'function': {
    'name': 'inspect_motion_interval',
    'description': (
        'Inspect a short interval of THIS existing video when sparse frames cannot distinguish '
        'retained travel from stance adjustment, a transient return, or event timing. Returns '
        'chronological images and forward/backward-checked image-motion evidence. '
        'Use only when it may resolve a consequential uncertainty; do not call for clearly visible '
        'endpoints. Maximum interval 2 seconds. This cannot reveal occluded surfaces, generate a '
        'new view, identify objects, measure metric 3D, prove contact forces or modify the world.'),
    'parameters': {'type': 'object', 'properties': {
        'start_sec': {'type': 'number'}, 'end_sec': {'type': 'number'},
        'question': {'type': 'string'}},
        'required': ['start_sec', 'end_sec', 'question'], 'additionalProperties': False}}}


def validate_interval(args, duration):
    import math
    start, end = args['start_sec'], args['end_sec']
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (start, end)):
        raise ValueError('Interval times must be finite numbers')
    if not 0 <= start < end <= duration+1e-8 or end-start > 2+1e-8:
        raise ValueError('Choose an existing video interval of at most two seconds')
    if not isinstance(args.get('question'), str) or not args['question'].strip():
        raise ValueError('State the consequential uncertainty this inspection addresses')
    return start, end


def review_arrival(config, video, contract, frames, output, reference=None, annotation=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    duration = frames[-1]['timestamp']
    selected = [frames[i] for i in sorted({0, len(frames)//2, len(frames)-1})]
    images = ([reference] if reference else []) + ([annotation] if annotation else [])
    images += [f['path'] for f in selected]
    messages = [{'role': 'system', 'content': (
        'You audit a coarse-world arrival using this actual video. You may request targeted temporal '
        'evidence, but you do not edit or accept a world. Distinguish intended target from observed '
        'endpoint. A projected target is an approximate annotation, not a new physical marker or '
        'metric ground truth. Judge retained whole-body/root travel against stable floor features, '
        'not just stance widening, apparent body motion, or dominant optical flow. Camera motion '
        'and perspective are uncertainties. Use coarse relative judgments; do not demand exact '
        'foot poses or visible proof of occluded contact points. Return JSON '
        '{arrival:SUPPORTED|CONTRADICTED|UNOBSERVED,retained_displacement:SUPPORTED|CONTRADICTED|UNOBSERVED,'
        'evidence:[string],uncertainties:[string],further_inspection_useful:boolean}. '
        'Use the optional tool only if it can resolve a consequential ambiguity. Tool image summaries '
        'are not independent of these same video pixels. Never claim continuous video viewing; '
        'you receive timestamped stills and computed motion, not direct video.')},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': json.dumps({'contract': contract,
                'image_order': 'optional accepted reference, optional annotated initial target, then video samples',
                'reference_included': bool(reference), 'annotation_included': bool(annotation),
                'sample_times': [f['timestamp'] for f in selected], 'video_last_timestamp': duration})},
            *[image_content(p) for p in images]]}]
    trace = []
    calls = 0
    try:
        for round_index in range(5):
            response = AgentClient(config).complete(messages, [INSPECT_TOOL] if calls < 3 else None)
            trace.append({'round': round_index, 'response': response})
            messages.append({k: v for k, v in response.items() if k in ('role', 'content', 'tool_calls')})
            if not response.get('tool_calls'):
                text = (response.get('content') or '').strip()
                if text.startswith('```'):
                    text = '\n'.join(text.splitlines()[1:-1])
                result = json.loads(text)
                for key in ('arrival', 'retained_displacement'):
                    if result.get(key) not in {'SUPPORTED', 'CONTRADICTED', 'UNOBSERVED'}:
                        raise ValueError('Invalid evidence verdict')
                result['world_modified'] = False
                result['inspection_calls'] = calls
                (output/'review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                return result
            image_messages = []
            for call in response['tool_calls']:
                try:
                    if calls >= 3:
                        raise ValueError('Inspection budget exhausted; report remaining uncertainty')
                    calls += 1
                    if call['function']['name'] != 'inspect_motion_interval':
                        raise ValueError('Unknown evidence tool')
                    args = json.loads(call['function']['arguments'])
                    start, end = validate_interval(args, duration)
                    evidence = inspect_motion(video, output/f'inspection_{calls:02d}', start, end, .125)
                    (output/f'inspection_{calls:02d}.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
                    summary = {k: evidence[k] for k in ('start_sec', 'end_sec', 'analysis_width',
                                                      'pair_count', 'measured_pairs', 'motion_peaks', 'limitations')}
                    summary['sample_times'] = [f['timestamp'] for f in evidence['frames']]
                    image_messages.append({'role': 'user', 'content': [
                        {'type': 'text', 'text': json.dumps({'inspection_call_id': call['id'],
                            'sample_times': summary['sample_times'], 'question': args['question']})},
                        *[image_content(f['path']) for f in evidence['frames']]]})
                except (ValueError, KeyError, TypeError) as exc:
                    summary = {'error': str(exc)}
                trace.append({'call_id': call['id'], 'result': summary})
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(summary)})
            messages.extend(image_messages)
        raise ValueError('Evidence reviewer did not finish within its bounded inspection budget')
    finally:
        (output/'tool_trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
