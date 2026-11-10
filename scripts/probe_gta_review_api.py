"""Check the configured review endpoint without logging credentials."""
import json
from pathlib import Path
import requests
import yaml


def main():
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())['agent']
    try:
        response = requests.post(cfg['base_url'].rstrip('/')+'/chat/completions',
            headers={'Authorization': 'Bearer '+cfg['api_key']},
            json={'model': cfg['model'], 'messages': [{'role':'user','content':'Return JSON only: {"ready":true}'}]},
            timeout=90)
        print(json.dumps({'status': response.status_code, 'body': response.json() if response.ok else 'request failed'}))
    except Exception as error:
        print(json.dumps({'error_type':type(error).__name__}))


if __name__ == '__main__':
    main()
