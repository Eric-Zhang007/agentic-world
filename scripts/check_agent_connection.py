"""Check the configured agent through the normal JSON client, without exposing credentials."""
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.action_planner import json_call


if __name__ == '__main__':
    config = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())['agent']
    result = json_call(config, 'Return JSON {"status":"OK"}.', {}, [])
    if result != {'status': 'OK'}:
        raise ValueError('Agent connectivity check returned unexpected data')
    print('Configured agent JSON request: OK')
