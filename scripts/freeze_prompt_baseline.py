"""Freeze a successful H3 replay as a regression baseline.

The manifest is intentionally small and immutable-by-convention: later
experiments must derive from it and may change only the declared prompt text.
"""
import argparse
import json
from pathlib import Path


def main(args):
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    command = metadata['command']
    prompt = args.prompt.read_text(encoding='utf-8')
    if prompt != command[command.index('--prompt') + 1]:
        raise ValueError('Prompt differs from the actual executed command')
    completion = json.loads(args.completion.read_text(encoding='utf-8'))
    review = json.loads(args.review.read_text(encoding='utf-8'))
    if completion.get('returncode') != 0:
        raise ValueError('Baseline generation did not complete successfully')
    if review.get('identity_consistency', {}).get('status') != 'SUPPORTED':
        raise ValueError('Baseline identity was not supported by its review')
    if args.completion.resolve().parent != args.metadata.resolve().parent:
        raise ValueError('Completion and executed record belong to different runs')
    if not args.review.resolve().is_relative_to(args.metadata.resolve().parent):
        raise ValueError('Review is not inside the recorded run')
    manifest = {
        'name': args.name,
        'source_metadata': str(args.metadata.resolve()),
        'source_prompt': str(args.prompt.resolve()),
        'command': command,
        'prompt': prompt,
        'allowed_variation': 'prompt text only; preserve all image/control inputs and renderer flags',
        'source_completion': str(args.completion.resolve()),
        'source_review': str(args.review.resolve()),
        'identity_assessment': review['identity_consistency'],
        'overall_assessment': review.get('overall'),
        'scope': 'Identity regression baseline only, not world acceptance',
        'world_commit': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print('Frozen exact executed command and prompt:', args.output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='reboot_live_identity_success')
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--completion', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
