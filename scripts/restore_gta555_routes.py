"""Reconstruct missing portable routing from the original source manifest."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    index = json.loads((a.source / 'data/index.json').read_text())
    by_id = {e['episode_id']: e for e in index['episodes']}
    original = json.loads((a.source / 'method/original_sources.json').read_text())
    episodes = []
    for s in original:
        rel = by_id[s['episode_id']]['path']
        episodes.append({'episode_id': s['episode_id'], 'group': s['group'],
                         'path': 'data/' + rel, 'spools': {
            kind: {'source': 'data/' + rel + ('/.spool/world_state.jsonl' if kind == 'world'
                    else '/.spool/agents/' + kind + '/client.jsonl'),
                   'index': 'data/record_indices/' + s['episode_id'] + '/' + kind + '.sqlite3'}
            for kind in s['spools']}})
    target = a.source / 'data/episodes.json'
    if target.exists():
        if json.loads(target.read_text()) != episodes:
            raise ValueError('Existing route differs; refusing overwrite')
    else:
        target.write_text(json.dumps(episodes, indent=2), encoding='utf-8')
    a.report.write_text(json.dumps({'episodes': len(episodes), 'derived': True,
        'inputs': ['data/index.json', 'method/original_sources.json'],
        'output': str(target), 'original_assembly_audit_restored': False}, indent=2))
    print(a.report.read_text())


if __name__ == '__main__':
    main()
