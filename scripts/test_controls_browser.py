"""Exercise the running local control preview and save desktop/mobile evidence."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright


def main():
    output=Path(__file__).resolve().parents[1]/'outputs/keyboard_controls/browser_review'
    output.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1280,'height':800})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://127.0.0.1:8766')
        page.wait_for_function("document.querySelector('#status').textContent.includes('就绪')")
        page.screenshot(path=str(output/'desktop.png'))
        pixels=page.evaluate("""() => {
          const c=document.querySelector('canvas'),g=c.getContext('webgl2');
          const p=new Uint8Array(c.width*c.height*4);g.readPixels(0,0,c.width,c.height,g.RGBA,g.UNSIGNED_BYTE,p);
          const colors=new Set();for(let i=0;i<p.length;i+=4*37)colors.add(`${p[i]},${p[i+1]},${p[i+2]}`);
          return colors.size;
        }""")
        assert pixels>50,pixels
        before=page.request.get('http://127.0.0.1:8766/state').json()
        page.locator('#start').click();page.keyboard.down('KeyW')
        page.wait_for_timeout(1200)
        page.keyboard.up('KeyW')
        canvas=page.locator('canvas').bounding_box()
        page.mouse.move(canvas['x']+canvas['width']/2,canvas['y']+canvas['height']/2)
        page.mouse.down();page.mouse.move(canvas['x']+canvas['width']/2+100,canvas['y']+canvas['height']/2+25,steps=12);page.mouse.up()
        page.wait_for_timeout(600);page.locator('#pause').click();page.wait_for_timeout(200)
        after=page.request.get('http://127.0.0.1:8766/state').json()
        assert after['ticks']>before['ticks']
        assert after['scene']['camera']['position']!=before['scene']['camera']['position']
        assert after['scene']['entities'][0]['position']!=before['scene']['entities'][0]['position']
        page.screenshot(path=str(output/'moved.png'))
        page.locator('#export').click()
        page.wait_for_function("/keyboard_controls[\\\\/].*\\d{8}/.test(document.querySelector('#status').textContent)",timeout=60000)
        plan=page.locator('#status').inner_text()
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(500)
        page.screenshot(path=str(output/'mobile.png'))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        result={'canvas_sampled_colors':pixels,'input_ticks':after['ticks'],
                'player_moved':True,'camera_moved':True,'exported_plan':plan,
                'desktop':[1280,800],'mobile':[390,844],'errors':errors}
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2));browser.close()


if __name__=='__main__':main()
