"""Probe native server Codex with the configured relay, without exposing its key."""
import os
from pathlib import Path
import subprocess
import yaml
import argparse

p=argparse.ArgumentParser()
p.add_argument('--prompt-file',type=Path)
p.add_argument('--events',type=Path)
p.add_argument('--resume')
p.add_argument('--proxy')
a=p.parse_args()

root=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((root/'configs/pi_relay.yaml').read_text())['agent']
env=dict(os.environ,RELAY_USER_TOKEN=cfg['api_key'])
if a.proxy:
 env.update(HTTP_PROXY=a.proxy, HTTPS_PROXY=a.proxy,
            http_proxy=a.proxy, https_proxy=a.proxy)
env['PATH']='/home/star/.nvm/versions/node/v22.23.2/bin:'+env.get('PATH','')
command=['codex','exec','--skip-git-repo-check','--json','-c','model_provider="relay"',
 '-c','model="gpt-6-astra"','-c','model_reasoning_effort="medium"',
 '-c','model_providers.relay.name="relay"',
 '-c','model_providers.relay.base_url="https://orouter.net/v1"',
 '-c','model_providers.relay.wire_api="responses"',
 '-c','model_providers.relay.env_key="RELAY_USER_TOKEN"',
 '-c','model_providers.relay.requires_openai_auth=false',
 '-c','model_providers.relay.supports_websockets=false',
 '-c','model_providers.relay.request_max_retries=0',
 '-c','model_providers.relay.stream_max_retries=0',
 '-c','features.enable_request_compression=false',
 '-c','analytics.enabled=false','-c','feedback.enabled=false',
 (a.prompt_file.read_text() if a.prompt_file else 'Reply exactly READY. Do not use tools.')]
if a.prompt_file:
 command.insert(2,'--dangerously-bypass-approvals-and-sandbox')
if a.resume:
 command.insert(2,'resume')
 command.insert(-1,a.resume)
out=a.events or root/'outputs/server_codex_probe.jsonl'
with out.open('x') as f:
 result=subprocess.run(command,env=env,cwd=root,stdout=f,stdin=subprocess.DEVNULL,timeout=3600 if a.prompt_file else 120)
print('Events saved:',out)
raise SystemExit(result.returncode)
