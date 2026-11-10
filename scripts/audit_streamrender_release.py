"""Reproducible release/interface audit; explicitly does not run H3 inference."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest


def audit(source, output):
    source = source.resolve()
    rows = [json.loads(line) for name in ('A','B')
            for line in (source/f'examples/requests_{name}.jsonl').read_text().splitlines() if line.strip()]
    palette = json.loads((source/'coding/threejs-racing/public/palette.json').read_text())
    config = json.loads((source/'configs/runtime.json').read_text())
    tests = []
    for pattern in ('test_protocol.py','test_http.py','test_examples.py'):
        run = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern],
                             cwd=source,text=True,capture_output=True,timeout=60)
        tests.append({'pattern':pattern,'returncode':run.returncode,'output':run.stdout+run.stderr})
    spec = importlib.util.spec_from_file_location('streamrender_release_server',source/'runtime/server.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    bridge = module.Bridge({'max_pending_frames':4,'output_queue_chunks':1})
    bridge.worker_ready = True
    geometry = []
    for width,height in [(1344,768),(1024,768),(832,480)]:
        try:
            session = bridge.create({'fps':24,'width':width,'height':height})
            session.closed = True
            geometry.append({'width':width,'height':height,'accepted':True})
        except ValueError as exc:
            geometry.append({'width':width,'height':height,'accepted':False,'reason':str(exc)})
    result = {
        'scope':'CPU configuration, catalog, HTTP and protocol audit. NO model inference or generalization score.',
        'examples':len(rows),'unique_semantic_videos':len({r['reference_video_path'] for r in rows}),
        'unique_reference_images':len({r['picture'] for r in rows}),
        'palette_categories':list(palette),
        'absent_gameplay_categories':[c for c in ('person','animal','weapon','held_object','projectile') if c not in palette],
        'backbone_dtype':config['inference']['models']['backbone']['runtime']['param_dtype'],
        'backbone_placement':config['inference']['models']['backbone']['placement'],
        'nfe':config['inference']['diffusion']['sampling_timesteps']['args']['num_sampling_steps'],
        'session_geometry_probes':geometry,'upstream_tests':tests,
        'interpretation':[
            'Missing palette classes mean outside demonstrated training vocabulary, not proof of impossible generation.',
            'Transport acceptance is not learned semantic support.',
            'Current output does not measure GTA or fantasy action success.',
            'Distillation causal effect on generalization requires a matched pre-distillation baseline.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='upstream_tests'},indent=2))
    if any(t['returncode'] for t in tests):
        raise SystemExit('One or more upstream tests failed; inspect saved results')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    audit(a.source,a.output)
