"""Verify portable packaging using existing accepted media, without changing releases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = args.output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    selected = {}
    for line in (args.mixture / 'train.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        selected.setdefault(row['task'], row)
    (inputs / 'roots.example.json').write_bytes((args.mixture / 'roots.example.json').read_bytes())
    for split in ('train', 'validation', 'test'):
        rows = selected.values() if split == 'train' else []
        (inputs / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    subprocess.run([sys.executable, '-m', 'scripts.package_gta_unified', '--mixture', str(inputs),
                    '--output', str(args.output / 'archive_check')], check=True)
    report = json.loads((args.output / 'archive_check/DELIVERY.json').read_text())
    assert report['records'] == len(selected)
    with tempfile.TemporaryDirectory(dir=args.output, prefix='unpacked_') as directory:
        with zipfile.ZipFile(args.output / 'archive_check/GTA_memory_training_unified.zip') as archive:
            archive.extractall(directory)
        subprocess.run([sys.executable, str(Path(directory) / 'example.py')], check=True)
    print(json.dumps({'passed': True, 'records': len(selected),
                      'scope': 'Packaging exercise using prior accepted examples; not a new delivery'}))


if __name__ == '__main__':
    main()
