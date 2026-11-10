"""Regression: exporting a paused aligned recording must not append braking ticks."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from playwright.sync_api import sync_playwright
from scripts.test_keyboard_controls import forward_scene

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    output=Path(directory)
    (output/'scene.json').write_text(json.dumps(forward_scene()))
    server=subprocess.Popen([sys.executable,str(root/'scripts/serve_controls.py'),
        '--scene',str(output/'scene.json'),'--output',str(output/'recordings'),'--port','0'],
        cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        line=server.stdout.readline();url=line.strip().split()[-1]
        assert url.startswith('http'),line
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='msedge',headless=True);page=browser.new_page()
            for _ in range(72):
                response=page.request.post(url+'/step',data={'keys':['w'],'mouse':[0,0]})
                assert response.ok,response.text()
            before=page.request.get(url+'/state').json()
            page.goto(url)
            page.wait_for_function("document.querySelector('#status').textContent.includes('就绪')")
            page.locator('#export').click()
            page.wait_for_function("document.querySelector('#status').textContent.includes('recordings')")
            after=page.request.get(url+'/state').json()
            assert after['ticks']==before['ticks']==72
            assert after['scene']==before['scene']
            action=json.loads(next((output/'recordings').glob('*/action.json')).read_text())
            assert len(action['inputs'])==72
            assert all(sample['keys']==['w'] for sample in action['inputs'])
            page.request.post(url+'/step',data={'keys':['w']})
            page.reload();page.wait_for_function("document.querySelector('#status').textContent.includes('就绪')")
            page.locator('#export').click();page.wait_for_timeout(200)
            assert page.request.get(url+'/state').json()['ticks']==73
            assert len(list((output/'recordings').glob('*/action.json')))==1
            browser.close()
        print('PASS: aligned export retains all 72 input ticks; unaligned pause does not append ticks')
    finally:
        server.terminate();server.wait(timeout=10)
