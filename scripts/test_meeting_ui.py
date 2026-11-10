"""Exercise real browser controls and submit one real H3 job on an isolated fork."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[1]
out=root/'outputs/meeting_demo/ui_review'
out.mkdir(exist_ok=True)
url='http://127.0.0.1:8768'
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    context=browser.new_context(viewport={'width':1280,'height':800},record_video_dir=str(out/'recording'))
    page=context.new_page();errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(url,timeout=120000)
    page.wait_for_function("document.querySelector('#status').textContent.includes('就绪')",timeout=120000)
    page.screenshot(path=str(out/'desktop.png'))
    colors=page.evaluate("""() => {const c=document.querySelector('canvas'),g=c.getContext('webgl2');
      const data=new Uint8Array(c.width*c.height*4);g.readPixels(0,0,c.width,c.height,g.RGBA,g.UNSIGNED_BYTE,data);
      const values=new Set();for(let i=0;i<data.length;i+=148)values.add(`${data[i]},${data[i+1]},${data[i+2]}`);return values.size;}""")
    assert colors>50,colors
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_timeout(500)
    page.screenshot(path=str(out/'mobile.png'))
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.set_viewport_size({'width':1280,'height':800})
    before=page.request.get(url+'/state').json()
    assert before['world_bound'] and not before.get('job'), 'Do not repeat an existing UI job'
    page.locator('#start').click();page.keyboard.down('KeyW')
    page.wait_for_function("Number(document.querySelector('#clock').textContent.match(/(\\d+) ticks/)[1])>=55",timeout=120000)
    canvas=page.locator('canvas').bounding_box()
    page.mouse.move(canvas['x']+canvas['width']/2,canvas['y']+canvas['height']/2)
    page.mouse.down();page.mouse.move(canvas['x']+canvas['width']/2+15,canvas['y']+canvas['height']/2,steps=5);page.mouse.up()
    page.locator('#render-keys').click()
    page.wait_for_function("document.querySelector('#render-keys').disabled",timeout=120000)
    page.keyboard.up('KeyW')
    after=page.request.get(url+'/state').json()
    assert after['job']['running']
    assert after['scene']['entities'][0]['position']!=before['scene']['entities'][0]['position']
    assert after['scene']['camera']['position']!=before['scene']['camera']['position']
    page.screenshot(path=str(out/'submitted.png'))
    assert not errors,errors
    result={'canvas_sampled_colors':colors,'world_bound':True,'input_ticks':after['ticks'],
            'real_job':after['job'],'player_moved':True,'camera_moved':True,
            'screenshots':['desktop.png','mobile.png','submitted.png'],'errors':errors}
    (out/'submission.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    context.close();browser.close()
