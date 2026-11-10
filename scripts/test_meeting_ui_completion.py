"""Check completed real job playback and the next input window, without another render."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

out=Path(__file__).resolve().parents[1]/'outputs/meeting_demo/ui_review'
url='http://127.0.0.1:8768'
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1280,'height':800})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    state=page.request.get(url+'/state').json()
    assert state['job'] and not state['job']['running'], 'Real job must finish first'
    receipt=state['job']['receipt']
    page.goto(url,timeout=120000)
    page.wait_for_function("/就绪|本段未接受/.test(document.querySelector('#status').textContent)",timeout=120000)
    if not receipt['accepted']:
        assert receipt['revision_before']==receipt['revision_after']
        assert receipt['history_before']==receipt['history_after']
        page.locator('#reset').click()
    accepted=page.request.get(url+'/state').json()
    assert accepted['revision']==receipt['revision_after']
    assert accepted['ticks']==0
    page.locator('#view-video').click()
    page.wait_for_function("document.querySelector('video').currentTime>.1",timeout=30000)
    page.screenshot(path=str(out/'accepted_video.png'))
    page.locator('#start').click();page.keyboard.down('KeyW')
    page.wait_for_function("Number(document.querySelector('#clock').textContent.match(/(\\d+) ticks/)[1])>=4",timeout=30000)
    page.keyboard.up('KeyW');page.locator('#pause').click()
    next_input=page.request.get(url+'/state').json()
    assert next_input['revision']==accepted['revision']
    assert next_input['time']>accepted['time']
    assert next_input['scene']['entities'][0]['position']!=accepted['scene']['entities'][0]['position']
    page.locator('#reset').click()
    page.locator('#action-prompt').fill('继续沿街向前走，保持镜头跟随，不要在段尾停下。')
    assert page.locator('#render-prompt').is_enabled()
    page.screenshot(path=str(out/'next_input.png'))
    assert not errors,errors
    result={'actual_job_receipt':receipt,'new_rollout_accepted':receipt['accepted'],
            'accepted_revision_loaded':accepted['revision'],
            'accepted_rgb_video_played':True,'next_input_advanced_from_accepted_state':True,
            'text_input_enabled':True,'text_ui_submit_tested':False,'browser_errors':errors}
    (out/'completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));browser.close()
