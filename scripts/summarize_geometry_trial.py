"""Inspect a Pi geometry trial without emitting embedded images or other runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_pi_run import inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary = inspect(args.events)
    last = summary.get('last_args') or {}
    if 'content' in last:
        last = dict(last, content='<file contents omitted>')
    summary['last_args'] = last
    operations = []
    for line in args.events.read_text(encoding='utf-8').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('type') == 'tool_execution_start' and event.get('toolName') == 'world':
            operations.append(event.get('args', {}))
    summary['world_calls'] = operations
    if args.output:
        summary['artifacts'] = [str(p.relative_to(args.output))
                                for p in args.output.rglob('*') if p.is_file()]
        report = args.output / 'report.json'
        if report.exists():
            summary['report'] = json.loads(report.read_text(encoding='utf-8'))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
