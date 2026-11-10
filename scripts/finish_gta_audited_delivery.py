"""Finish manifests, packaging and authorized HF upload after a complete audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root
    audit=root/'scene_memory_audit'
    status=root/'AUDITED_DELIVERY_STATUS.json'
    def save(stage,**details):
        status.write_text(json.dumps({'stage':stage,'updated_at':time.time(),**details},indent=2))
    def run(stage,module,args):
        save(stage,status='running')
        with (root/(stage+'.log')).open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','-m',module,*map(str,args)],check=True,
                stdout=log,stderr=subprocess.STDOUT)
    try:
        while True:
            if (audit/'error.json').exists():raise ValueError('Visual audit stopped on an API or evidence error')
            if (audit/'run.json').exists() and (audit/'status.json').exists():
                try:
                    progress=json.loads((audit/'status.json').read_text())
                except (json.JSONDecodeError, PermissionError):
                    time.sleep(1)
                    continue
                if progress['completed_batches']==progress['total_batches']:break
                save('visual_audit',status='running',progress=progress)
            time.sleep(15)
        sources=[root/'available_source_training',root/'additional_local_training']
        run('additional_metadata','scripts.enrich_gta_training_metadata',[
            '--root',sources[1],'--indices',root/'record_adapter','--output',root/'additional_source_metadata'])
        mixture=root/'audited_mixture';delivery=root/'unified_delivery'
        run('assemble_audited','scripts.assemble_gta_audited_mixture',[
            '--roots',*sources,'--audit',audit,'--old-mixture','D:/GTA_memory_mixture_20261019',
            '--split-map','configs/gta_local_episode_splits.json','--output',mixture])
        run('package_audited','scripts.package_gta_unified',[
            '--mixture',mixture,'--output',delivery,'--auxiliary-roots',
            root/'available_source_metadata',root/'additional_source_metadata'])
        save('hf_upload',status='running')
        linux_delivery='/mnt/'+delivery.drive[0].lower()+'/'+ '/'.join(delivery.parts[1:])
        with (root/'hf_audited_upload.log').open('a',encoding='utf-8') as log:
            subprocess.run(['wsl.exe','-e','python3','/mnt/c/Users/zjc/omgd/agentic_world/scripts/upload_gta_audited.py',
                '--delivery',linux_delivery],stdout=log,stderr=subprocess.STDOUT,check=True)
        save('complete',status='UPLOADED_AND_REMOTE_SIZE_VERIFIED',delivery=str(delivery))
    except Exception as error:
        save('failed',status='NOT_UPLOADED_OR_INCOMPLETE',error_type=type(error).__name__,reason=str(error))
        raise


if __name__=='__main__':main()
