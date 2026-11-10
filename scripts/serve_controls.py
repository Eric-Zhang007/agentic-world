"""Local keyboard preview. Exported recordings enter the normal checked plan path."""
import argparse
from copy import deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.pi.world_tools import read,save,dispatch
from whitebox.controls import Controller,input_sample
from whitebox.motion import simulate
from scripts.reconcile_whitebox_video import align_action


class Session:
    def __init__(self,scene,actor,output,db=None,first_frame=None,gpu='2'):
        self.original=scene;self.actor=actor;self.output=output
        self.db=str(Path(db).resolve()) if db else None
        self.first_frame=first_frame;self.gpu=gpu;self.job=None;self.job_dir=None
        self.revision=0;self.start_time=0.
        self.last_video=None
        self.reset()

    def reset(self):
        if self.job and self.job.poll() is None:raise ValueError('A rollout is still running')
        if self.db:
            state=dispatch('status',{'db':self.db})
            self.original=state['scene'];self.revision=state['revision'];self.start_time=state['tick']
            if state.get('last_frame'):self.first_frame=state['last_frame']
            from state.db import WorldDB
            db=WorldDB(self.db)
            try:self.last_video=db.get_meta('whitebox_last_video')
            finally:db.connection.close()
        self.controller=Controller(self.original,self.actor,self.start_time)
        self.inputs=[]

    def state(self):
        job=None
        if self.job:
            code=self.job.poll();receipt=self.job_dir/'job_exit.json'
            job={'running':code is None,'returncode':code,'directory':str(self.job_dir),
                 'receipt':read(receipt) if receipt.exists() else None}
        return {'scene':self.controller.scene,'ticks':len(self.inputs),
                'time':self.controller.time,'events':self.controller.events[-8:],
                'world_bound':bool(self.db),'revision':self.revision,'job':job,
                'accepted_video_available':bool(self.last_video)}

    def step(self,sample):
        if self.job and self.job.poll() is None:raise ValueError('Wait for video review before the next input window')
        sample=input_sample(sample)
        # A rejected tick must not leave a partially advanced preview behind.
        backup=deepcopy(self.controller)
        try:self.controller.step(sample)
        except Exception:
            self.controller=backup
            raise
        self.inputs.append(sample)
        return self.state()

    def export(self):
        action={'type':'controls','actor_id':self.actor,'goal':'Replay recorded keyboard and mouse input',
                'inputs':self.inputs}
        action,alignment=align_action(action)
        timeline,end,report=simulate(self.original,action,self.start_time)
        if end!=self.controller.scene:raise ValueError('Recording replay differs from interactive state')
        directory=self.output/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        directory.mkdir(parents=True,exist_ok=False)
        if self.db:
            current=dispatch('status',{'db':self.db})
            if current['revision']!=self.revision or current['scene']!=self.original:
                raise ValueError('World changed; refresh accepted state before recording again')
            pending=current.get('pending_action')
            if pending and not pending['action_complete']:action['goal']=pending['continuation_goal']
            save(directory/'input.json',action)
            result=dispatch('plan',{'db':self.db,'action':str(directory/'input.json'),
                'output':str(directory/'plan')})
            return {'plan':result['plan'],'frames':result['video_frames'],'world_bound':True}
        for name,value in [('scene',self.original),('action',action),('timeline',timeline),
                ('endpoint',end),('diagnostics',report),('status',{'revision':0,'world_start':0.,
                    'status':'PLANNED_NOT_COMMITTED','alignment':alignment,'source':'keyboard preview'})]:
            save(directory/(name+'.json'),value)
        return {'plan':str(directory),'frames':len(timeline['frames'])+1,
                'message':'Preview exported. To use an existing world, run world plan with this action and that DB.'}

    def rollout(self,mode,goal=''):
        if not self.db:raise ValueError('Start this server with --db and an actual RGB first frame')
        if self.job and self.job.poll() is None:raise ValueError('A rollout is already running')
        if not Path('/home/star/agentic-world-comfy/.venv/bin/python').is_file():
            raise ValueError('Launch the world-bound control server on the H3 host')
        if not self.first_frame or not Path(self.first_frame).is_file():raise ValueError('Actual starting RGB required')
        current=dispatch('status',{'db':self.db})
        if current['revision']!=self.revision:raise ValueError('Refresh the changed world before submitting')
        if mode not in ('keyboard','prompt'):raise ValueError('Unknown input mode')
        if mode=='prompt' and not goal.strip():raise ValueError('Action prompt required')
        plan=self.export()['plan'] if mode=='keyboard' else None
        directory=self.output/('live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        directory.mkdir(parents=True,exist_ok=False)
        request={'db':self.db,'revision':self.revision,'mode':mode,'goal':goal,
                 'action':str(Path(plan)/'action.json') if plan else None,
                 'first_frame':str(self.first_frame),'gpu':self.gpu}
        save(directory/'request.json',request)
        with (directory/'launcher.log').open('x') as log:
            self.job=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_interactive_rollout.py'),
                '--request',str(directory/'request.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        self.job_dir=directory
        return {'job':str(directory),'status':'RUNNING'}



def main():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--scene',type=Path);source.add_argument('--db',type=Path)
    p.add_argument('--first-frame',type=Path);p.add_argument('--gpu',default='2')
    p.add_argument('--actor',default='player')
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/keyboard_controls')
    a=p.parse_args();session=Session(read(a.scene) if a.scene else None,a.actor,a.output,a.db,a.first_frame,a.gpu)
    web=ROOT/'harness/controls'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,value,status=200):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/state':return self.respond(session.state())
            paths={'/':(web/'index.html','text/html; charset=utf-8'),
                   '/app.js':(web/'app.js','text/javascript'),
                   '/three.js':(web/'vendor/three.module.js','text/javascript'),
                   '/three.core.js':(web/'vendor/three.core.js','text/javascript'),
                   '/catalog':(ROOT/'assets/gta/catalog.json','application/json')}
            geometry=session.original.get('static_geometry',{}).get('path')
            if geometry:paths['/geometry']=(Path(geometry),'application/octet-stream')
            if session.last_video:paths['/accepted-video']=(Path(session.last_video),'video/mp4')
            request_path=self.path.split('?',1)[0]
            if request_path not in paths:return self.respond({'error':'Not found'},404)
            path,mime=paths[request_path]
            if not path.is_file():return self.respond({'error':'Missing asset: '+str(path)},404)
            data=path.read_bytes();self.send_response(200);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_POST(self):
            try:
                value=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}')
                if self.path=='/step':result=session.step(value)
                elif self.path=='/reset':session.reset();result=session.state()
                elif self.path=='/export':result=session.export()
                elif self.path=='/rollout':result=session.rollout(value['mode'],value.get('goal',''))
                else:return self.respond({'error':'Not found'},404)
                self.respond(result)
            except (ValueError,KeyError,TypeError) as e:self.respond({'error':str(e)},400)

    server=HTTPServer(('127.0.0.1',a.port),Handler)
    print(f'Keyboard preview http://127.0.0.1:{server.server_port}',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
