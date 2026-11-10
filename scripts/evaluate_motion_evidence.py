"""Read-only real-video comparison; never commits a world or changes old reviews."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.motion_evidence import inspect_motion


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = inspect_motion(args.video, args.output/'motion', args.start, args.end)
    save(args.output/'motion.json', evidence)
    if not args.config:
        return
    import yaml
    from agent.action_planner import json_call
    config = yaml.safe_load(args.config.read_text())['agent']
    frames = evidence['frames']
    sparse = [frames[i] for i in sorted({round(i*(len(frames)-1)/16) for i in range(17)})]
    system = ('Audit a coarse world action using visible evidence, not intended outcomes. '
              'Return JSON: {observed_start:string, observed_end:string, '
              'action_complete:true|false|null, event_interval_sec:[number,number]|null, '
              'uncertainties:[string], evidence:[string], requested_inspection:string|null}. '
              'Question: did the tiger reach a stable low alert posture supported by both '
              'forepaws and both feet, without falling or being defeated? Distinguish crouching '
              'with hands above ground from supported low guard. Do not infer contact forces. '
              'These are ordered stills, not a directly viewed video.')
    baseline = json_call(config, system, {'sample_times':[f['timestamp'] for f in sparse]},
                         [f['path'] for f in sparse])
    save(args.output/'sparse_review.json', baseline)
    # A local chronological window adds temporal detail without filling context with the full video.
    selected = frames[:24]
    enriched = json_call(config, system,
                         {'sample_times':[f['timestamp'] for f in selected],
                          'whole_interval_boundary_times':[sparse[0]['timestamp'], sparse[-1]['timestamp']],
                          'image_order':'first/last interval boundaries, then chronological local window',
                          'motion_peaks':evidence['motion_peaks'], 'limitations':evidence['limitations']},
                         [sparse[0]['path'], sparse[-1]['path']]+[f['path'] for f in selected])
    save(args.output/'motion_review.json', enriched)
    save(args.output/'comparison.json', {
        'sparse_complete': baseline.get('action_complete'),
        'motion_complete': enriched.get('action_complete'),
        'world_modified': False,
        'caveat':'Pilot with differing frame counts; not a controlled accuracy benchmark. '
                 'It cannot isolate the benefit of optical flow from denser sampling.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float)
    parser.add_argument('--config', type=Path)
    main(parser.parse_args())
