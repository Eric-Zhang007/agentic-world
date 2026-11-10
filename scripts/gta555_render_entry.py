"""Call original snapshot/render code with a locally discovered CPU EGL device."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-egl', action='store_true')
    p.add_argument('--source', type=Path)
    p.add_argument('--episode')
    p.add_argument('--agent')
    p.add_argument('--segment')
    p.add_argument('--destination', type=Path)
    a = p.parse_args()
    if a.probe_egl:
        import moderngl
        context = moderngl.create_standalone_context(backend='egl')
        print(json.dumps({'renderer': context.info['GL_RENDERER'], 'version': context.info['GL_VERSION']}))
        context.release()
        return
    if a.destination.exists():
        raise ValueError('Destination already exists')
    if a.destination.resolve().is_relative_to(a.source.resolve()):
        raise ValueError('Destination must be outside source')
    sys.path.insert(0, str(a.source / 'tools'))
    from reproduce import prepare
    command, env = prepare(a.episode, a.agent, a.segment, a.destination)
    probes = []
    for device in range(16):
        env['GLCONTEXT_DEVICE_INDEX'] = str(device)
        probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-egl'],
                               env=env, capture_output=True, text=True, timeout=30)
        if probe.returncode == 0:
            info = json.loads(probe.stdout)
            probes.append({'device': device, **info})
            if 'llvmpipe' in info['renderer'].lower():
                break
        else:
            probes.append({'device': device, 'error': probe.stderr[-1500:]})
            if 'but found' in probe.stderr:
                raise RuntimeError('No llvmpipe device: ' + json.dumps(probes))
    else:
        raise RuntimeError('No CPU EGL device found: ' + json.dumps(probes))
    (a.destination / 'runtime_environment.json').write_text(json.dumps({
        'device_index': device, 'renderer': info, 'probes': probes,
        'reason': 'Original device index 8 is host-specific; preserve CPU Mesa rendering'}, indent=2))
    with (a.destination / 'render.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report_path = a.destination / 'reproduction.json'
    report = json.loads(report_path.read_text())
    report['rendered'] = True
    report['runtime_environment'] = 'runtime_environment.json'
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
