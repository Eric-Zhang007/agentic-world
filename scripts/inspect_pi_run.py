"""Summarize Pi progress without printing embedded image payloads."""
import argparse
import json
from pathlib import Path
import time


def inspect(path):
    calls, completed, errors = [], set(), []
    last_text = None
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get('type')
            if kind == 'tool_execution_start':
                calls.append(event)
            elif kind == 'tool_execution_end':
                completed.add(event.get('toolCallId'))
                if event.get('isError'):
                    errors.append(event)
            elif kind == 'message_end' and event.get('message', {}).get('role') == 'assistant':
                last_text = '\n'.join(c.get('text', '') for c in event['message'].get('content', []) if c.get('type') == 'text')
    last = calls[-1] if calls else {}
    result = dict(calls=len(calls), errors=len(errors),
                  last_tool=last.get('toolName'), last_args=last.get('args'),
                  waiting_on_tool=bool(last) and last.get('toolCallId') not in completed,
                  log_age_seconds=round(time.time()-path.stat().st_mtime, 1),
                  last_assistant=last_text)
    if errors:
        result['last_error'] = [c.get('text') for c in errors[-1].get('result', {}).get('content', []) if c.get('type') == 'text']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.events), ensure_ascii=False, indent=2))
    for log in args.events.parent.rglob('generation.log'):
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-600))
            print(str(log)+':\n'+stream.read().decode('utf-8', errors='replace'))
