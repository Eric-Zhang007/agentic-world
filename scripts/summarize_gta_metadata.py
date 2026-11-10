"""Build compact sampling annotations from recorded auxiliary observations."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def active(value):
    if isinstance(value, dict):
        return value.get('pressed') is True or value.get('held') is True
    return value is True


def summarize(payload):
    frames = payload['target_samples']
    actions, modes, weather = set(), set(), set()
    states = defaultdict(list)
    fields = Counter()
    for frame in frames:
        world = frame.get('world_record', {})
        for name in ('agents', 'vehicles', 'peds', 'objects', 'relation_edges', 'entity_lifecycle_events'):
            if name in world:
                fields[name + '_recorded_samples'] += 1
                fields[name + '_nonempty_samples'] += bool(world[name])
        environment = world.get('environment', {})
        if environment.get('weather') is not None:
            weather.add(str(environment['weather']))
        # Lower bracket is used for sampled state changes; discrete inputs are
        # retained across both brackets without inventing interpolated actions.
        clients = frame.get('client_brackets', [])
        if clients:
            state = clients[0].get('agent_state', {})
            if state.get('available') is not False:
                for name in ('health', 'alive', 'in_vehicle'):
                    if name in state and state[name] is not None:
                        states[name].append(state[name])
        for client in clients:
            action = client.get('action', {})
            if action.get('mode') is not None:
                modes.add(str(action['mode']))
            for name in ('jump', 'sprint', 'crouch', 'interact', 'enter_exit_vehicle', 'primary_action', 'secondary_action'):
                if active(action.get(name)):
                    actions.add(name)
            for name in ('move_forward', 'move_right'):
                value = action.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) > 0:
                    actions.add('movement_input')
    return {'sample_id': payload['sample_id'], 'episode_id': payload['episode_id'],
            'sampled_active_inputs': sorted(actions), 'recorded_modes': sorted(modes),
            'recorded_weather': sorted(weather), 'record_coverage': dict(fields),
            'sampled_state_changes': {k: any(a != b for a,b in zip(v,v[1:])) for k,v in states.items()},
            'scope': 'Five sampled target times; observed input/state, not executed-action or visibility labels'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(Counter)
    temporary = a.output / 'sampling_annotations.tmp'
    count = 0
    with temporary.open('w', encoding='utf-8') as out:
        for path in sorted(a.metadata.glob('ep*.json')):
            row = summarize(json.loads(path.read_bytes()))
            sample = json.loads((a.training_root / 'training/samples' / row['sample_id'] / 'sample.json').read_bytes())
            row['review_status_at_scan'] = sample['training_status']
            row['passage_kind'] = sample['reference']['kind']
            row['reference_age_sec'] = sample['reference']['age_sec']
            row['reference_angle_deg'] = sample['reference'].get('view_angle_deg')
            group = 'sampled_review_passed' if row['review_status_at_scan'] == 'AUTO_REFERENCE_WITH_REVIEWED_TARGET_QA' else 'not_accepted'
            totals[group]['samples'] += 1
            for category in ('sampled_active_inputs','recorded_modes','recorded_weather'):
                for value in row[category]:
                    totals[group][category + ':' + value] += 1
            for key,changed in row['sampled_state_changes'].items():
                totals[group]['state_recorded:' + key] += 1
                totals[group]['state_changed:' + key] += changed
            out.write(json.dumps(row) + '\n')
            count += 1
            if count % 100 == 0:
                print('SUMMARIZED', count, flush=True)
    temporary.replace(a.output / 'sampling_annotations.jsonl')
    report = {'samples':count, 'groups':{k:dict(v) for k,v in totals.items()},
              'note':'Overlapping labels; not mixture percentages. Refresh acceptance before final sampling.'}
    (a.output / 'coverage.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
