"""Replay a saved world render without history; never commits a world state."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from renderer.comfy_h3 import ComfyH3Renderer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    requests = []
    for line in a.events.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        args = event.get('args', {})
        if event.get('type') == 'tool_execution_start' and event.get('toolName') == 'world' and args.get('operation') == 'render':
            requests.append(args['args'])
    if len(requests) != 1:
        raise ValueError('Expected exactly one source world render for this outbound ablation')
    source = requests[0]
    request = dict(source['request'])
    if request.get('history_pair'):
        raise ValueError('Source should use DB-selected history, not manually supplied images')
    source_output = Path(request['output_dir'])
    receipt = json.loads((source_output/'generation_metadata.json').read_text())
    if receipt.get('returncode') != 0:
        raise ValueError('Source generation did not complete')
    if not json.loads((source_output/'memory_selection.json').read_text())['candidates']:
        raise ValueError('Source did not use historical memory')
    a.output.mkdir(parents=True, exist_ok=False)
    request['output_dir'] = str(a.output)
    record = dict(source=str(source_output), renderer=source['renderer'], request=request,
                  difference='No historical image; tool-owned prompt removes historical bindings and renumbers current RGB.',
                  world_committed=False)
    (a.output/'ablation_request.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    video = ComfyH3Renderer(**source['renderer']).render(**request)
    print(video)


if __name__ == '__main__':
    main()
