"""Local, offline RGB-to-geometry proposal orchestration; never commits a world."""
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {
    'image', 'output', 'reconstruction', 'perception', 'frame', 'python',
    'reconstruction_python', 'perception_python', 'gpu', 'device', 'text',
    'threshold', 'scale', 'scale_source', 'actor_height_prior',
    'confidence_quantile', 'plane_threshold', 'timeout', 'godot', 'preview_scale', 'use_display',
}


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _file(value):
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'Expected file: {path}')
    return path


def _same_rgb(image, perception):
    from PIL import Image

    with Image.open(image) as a, Image.open(perception.parent / 'input.png') as b:
        a, b = a.convert('RGB'), b.convert('RGB')
        if a.size != b.size or a.tobytes() != b.tobytes():
            raise ValueError('Requested RGB differs from perception input.png; refusing cached masks')
        return list(a.size)


def propose(args: Mapping) -> dict:
    """Return JSON-safe candidate paths, evidence, failures and stage timings.

    Required: image. Optional reconstruction (NPZ), perception (result JSON),
    frame (cached NPZ index), output (NEW directory), python and stage-specific
    Python executables. All paths belong to the executing host. See docs.
    Operational/validation failures return success=False, never auto-accept.
    """
    started = time.perf_counter()
    result = dict(success=False, status='FAILED', accepted=False,
                  world_merge_ready=False, database_written=False,
                  scene_path=None, preview=None, review_images=[], evidence={}, stages={},
                  warnings=['Geometry is inferred; hidden surfaces and scale are not measured.',
                            'RGB consistency is not independent geometric or visual GT validation.'])
    output = None
    stage = 'validation'
    active = time.perf_counter()

    def finish_stage(status, **extra):
        result['stages'].setdefault(stage, {}).update(
            status=status, seconds=time.perf_counter() - active, **extra)

    def run(name, command):
        nonlocal stage, active
        stage, active = name, time.perf_counter()
        log = output / (name + '.log')
        result['stages'][name] = dict(command=[str(x) for x in command], log=str(log))
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
        result['stages'][name]['environment'] = {
            key: env[key] for key in ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES',
                                     'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
        with log.open('w', encoding='utf-8') as stream:
            proc = subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
        result['stages'][name]['returncode'] = proc.returncode
        if proc.returncode:
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 6000))
                detail = stream.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'{name} exited {proc.returncode}: {detail}')
        finish_stage('completed')

    try:
        if not isinstance(args, Mapping):
            raise ValueError('args must be a mapping')
        unknown = set(args) - OPTIONS
        if unknown:
            raise ValueError(f'Unknown options: {sorted(unknown)}')
        image = _file(args['image'])
        gpu = str(args.get('gpu', '2'))
        if not gpu.isdecimal():
            raise ValueError('gpu must be one physical GPU index')
        if args.get('device', 'cuda:0') not in ('cuda:0', 'cuda', 'cpu'):
            raise ValueError('device must be cuda:0 (selected physical gpu) or cpu')
        frame = args.get('frame', 0)
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError('frame must be a nonnegative integer')
        if frame and not args.get('reconstruction'):
            raise ValueError('Nonzero frame requires cached reconstruction')
        preview_scale = args.get('preview_scale', 4)
        if type(preview_scale) is not int or not 1 <= preview_scale <= 4:
            raise ValueError('preview_scale must be an integer in [1,4]')
        timeout = float(args.get('timeout', 600))
        for key, default in [('timeout', 600), ('scale', 1), ('plane_threshold', .02),
                             ('actor_height_prior', 1.7)]:
            value = float(args.get(key, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        for key, default in [('threshold', .25), ('confidence_quantile', .2)]:
            value = float(args.get(key, default))
            if not 0 <= value < 1:
                raise ValueError(f'{key} must be in [0,1)')
        reconstruction = _file(args['reconstruction']) if args.get('reconstruction') else None
        perception = _file(args['perception']) if args.get('perception') else None
        proposed_output = Path(args.get('output') or ROOT / 'outputs' / 'geometry_proposals' / uuid.uuid4().hex).expanduser().resolve()
        proposed_output.mkdir(parents=True, exist_ok=False)
        output = proposed_output
        result['output'] = str(output)
        result['evidence']['pipeline_report'] = str(output / 'proposal.json')
        result['input'] = dict(image=str(image), frame=frame)
        python = str(args.get('python') or sys.executable)
        finish_stage('completed')

        if reconstruction is None:
            reconstruction = output / 'reconstruction.npz'
            run('reconstruction', [args.get('reconstruction_python') or python, '-B',
                ROOT / 'scripts/reconstruct_vggt.py', image, '--output', reconstruction,
                '--gpu', gpu, '--local-files-only'])
            _file(reconstruction)
        else:
            result['stages']['reconstruction'] = dict(status='cached', seconds=0.)
        result['evidence']['reconstruction'] = str(reconstruction)
        if perception is None:
            perception = output / 'perception' / 'result.json'
            run('perception', [args.get('perception_python') or python, '-B',
                ROOT / 'scripts/probe_perception.py', '--image', image,
                '--output', perception.parent, '--device', args.get('device', 'cuda:0'),
                '--text', args.get('text', 'person. car. building. road. sidewalk. tree. pole. wall.'),
                '--threshold', args.get('threshold', .25)])
            _file(perception)
        else:
            result['stages']['perception'] = dict(status='cached', seconds=0.)
        result['evidence']['perception'] = str(perception)
        stage, active = 'rgb_and_masks', time.perf_counter()
        size = _same_rgb(image, perception)
        observations = _json(perception)['observations']
        categories = [item['category'].strip().rstrip('.') for item in observations]
        result['evidence']['requested_rgb_matches_perception'] = True
        result['evidence']['image_size'] = size
        result['evidence']['categories'] = categories
        ground_labels = {'road', 'ground', 'sidewalk'}
        accepted_ground = sorted(set(categories) & ground_labels)
        mixed_ground = sorted({c for c in categories if c not in ground_labels
                               and set(c.split()) & ground_labels})
        result['evidence']['ground_segmentation'] = {
            'exact_ground_categories': accepted_ground,
            'mixed_labels_not_used_as_ground': mixed_ground,
            'exact_road_detected': 'road' in categories,
        }
        if 'road' not in categories:
            result['warnings'].append('No exact road category detected; road surface reconstruction is not established.')
        if mixed_ground:
            result['warnings'].append('Mixed ground labels do not establish ground; inspect compiler category_mapping and source_candidates for disposition: '
                                      + ', '.join(mixed_ground))
        if not accepted_ground:
            raise ValueError('No road/ground/sidewalk mask detected; ground fitting unavailable. No fallback invented.')
        finish_stage('completed')

        candidate = output / 'candidate'
        command = [python, '-B', ROOT / 'scripts/compile_geometry_assets.py',
                   '--reconstruction', reconstruction, '--perception', perception,
                   '--frame', frame, '--output', candidate]
        for key in ('scale', 'scale_source', 'actor_height_prior',
                    'confidence_quantile', 'plane_threshold'):
            if key in args:
                command += ['--' + key.replace('_', '-'), args[key]]
        run('compile', command)
        scene_path = _file(candidate / 'scene.json')
        scene, report = _json(scene_path), _json(candidate / 'report.json')
        result['scene_path'] = str(scene_path)
        result['evidence'].update(compiler_report=str(candidate / 'report.json'),
            source_candidates=str(_file(candidate / 'source_candidates.json')),
            reconstruction_rgb_check='passed existing compiler preprocessing/RGB check',
            ground_surface=report.get('ground_surface'),
            facade_regions=report.get('facade_regions', []),
            scale_source=report.get('scale_source'), entity_count=len(scene['entities']))
        if any(item.get('coverage', 1) < 1 for item in report.get('facade_regions', [])):
            result['warnings'].append('Facade coverage incomplete; source_candidates preserves unfitted proposals, not accepted collision.')
        rejected_support = [item for item in report.get('ground_surface', {}).get('completion_evidence', [])
                            if item.get('status') == 'REJECTED']
        if rejected_support:
            result['warnings'].append(f'{len(rejected_support)} ground completion requests rejected; inspect ground_surface.completion_evidence. No support invented.')
        intrinsics = scene['camera']['intrinsics']
        preview = output / 'preview'
        stage, active = 'preview', time.perf_counter()
        width, height = (int(intrinsics[k]) * preview_scale for k in ('width', 'height'))
        if min(width, height) < 64 or max(width, height) > 4096:
            raise ValueError('Preview dimensions outside renderer [64,4096]; choose a smaller preview_scale')
        result['evidence']['preview_size'] = [width, height]
        command = [python, '-B', '-m', 'whitebox.godot.render', '--scene', scene_path,
                   '--output', preview, '--width', width,
                   '--height', height, '--timeout', max(1, timeout - 5)]
        if args.get('godot'):
            command += ['--godot', args['godot']]
        if args.get('use_display', False):
            command += ['--use-display']
        run('preview', command)
        manifest = _json(preview / 'manifest.json')
        images = []
        for item in manifest['images']:
            path = (preview / item['file']).resolve()
            if not path.is_relative_to(preview):
                raise ValueError('Preview manifest path escapes output directory')
            images.append(str(_file(path)))
        if len(images) < 4:
            raise ValueError('Missing required preview views')
        review_images = [str((preview / name).resolve()) for name in ('camera.png', 'overview.png')]
        if not all(path in images for path in review_images):
            raise ValueError('Preview manifest must contain camera.png and overview.png for Pi review')
        result['preview'] = dict(manifest=str(preview / 'manifest.json'), images=images)
        result['review_images'] = review_images
        result.update(success=True, status='PROPOSAL_READY')
    except Exception as exc:
        finish_stage('failed')
        result['error'] = dict(stage=stage, type=type(exc).__name__, message=str(exc))
    result['timing_seconds'] = {name: info['seconds'] for name, info in result['stages'].items()}
    result['timing_seconds']['total'] = time.perf_counter() - started
    if output is not None:
        (output / 'proposal.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--args', required=True, help='JSON file containing propose arguments')
    result = propose(_json(parser.parse_args().args))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['success'] else 1)
