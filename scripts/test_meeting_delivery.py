"""Verify the offline presentation opens, plays every video and fits both viewports."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root=Path.home()/'Desktop/Agentic_World_Demo'
evidence=Path(__file__).resolve().parents[1]/'outputs/meeting_demo/delivery_review'
evidence.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((root/'index.html').as_uri())
    page.wait_for_function("document.querySelector('video').readyState>=1")
    page.screenshot(path=str(evidence/'desktop.png'))
    movies=[]
    for button in page.locator('button[data-src]').all():
        source=button.get_attribute('data-src');button.click()
        page.wait_for_function("document.querySelector('video').currentTime>.15",timeout=15000)
        movies.append(page.locator('#player').evaluate('(v)=>({src:v.getAttribute("src"),duration:v.duration,width:v.videoWidth,height:v.videoHeight})'))
        assert movies[-1]['src']==source
        assert movies[-1]['width']>0
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_timeout(300)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(evidence/'mobile.png'))
    assert page.locator('img').evaluate_all('(items)=>items.every(x=>x.complete && x.naturalWidth>0)')
    assert not errors,errors
    result={'offline':True,'playable_videos':movies,'desktop':[1440,1000],
            'mobile':[390,844],'images_loaded':True,'errors':errors}
    (evidence/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));browser.close()
