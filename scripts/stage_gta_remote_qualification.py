"""Stage metadata and source Parquet files only; preserve local video originals."""
import json
from pathlib import Path
import sys
import time
import zipfile


def main():
    root=Path('E:/GTA_memory_release_20261018');project=Path(__file__).resolve().parents[1]
    ledger=[json.loads(s) for s in (root/'source_ledger.jsonl').read_text().splitlines()]
    pending=[r for r in ledger if r['status']=='PENDING']
    output=root/'remote_metadata_input.zip'
    if not output.exists():
        with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            archive.write(root/'source_ledger.jsonl','source_ledger.jsonl')
            completed=[json.loads(p.read_text())['source']['source_manifest'] for p in (root/'qualification').glob('*/summary.json')]
            archive.writestr('already_qualified.json',json.dumps(completed))
            episodes={r['record_index']:Path(r['episode_root']) for r in pending}
            for index,source in episodes.items():
                archive.write(root/index,'indexes/'+index)
                files=list((source/'agents').glob('*/camera.parquet'))+[source/'server/world_state.parquet',source/'server/entities.parquet']
                for path in files:archive.write(path,'sources/'+Path(index).stem+'/'+path.relative_to(source).as_posix(),compress_type=zipfile.ZIP_STORED)
            for row in pending:
                segment=Path(row['source_manifest']).parent;source=Path(row['episode_root'])
                for name in ('manifest.json','frames.jsonl'):
                    path=segment/name
                    archive.write(path,'sources/'+Path(row['record_index']).stem+'/'+path.relative_to(source).as_posix())
            for relative in ['memory/gta_release.py','memory/gta_validation.py','memory/gta_visibility.py',
                             'scripts/build_gta_release_index.py','scripts/prepare_gta_object_memory.py','scripts/gta_remote_qualify.py']:
                archive.write(project/relative,'code/'+relative)
        print('ARCHIVE',output.stat().st_size,flush=True)
    sys.path.insert(0,str(project.parent))
    from server_exec import connect
    client=connect();remote='/mnt/hdd/gta_release_20261018'
    try:
        _,stdout,stderr=client.exec_command('mkdir -p '+remote)
        if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode())
        start=time.monotonic();last=[start]
        def progress(done,total):
            now=time.monotonic()
            if now-last[0]>=15:
                print('UPLOAD',done,total,round(done/max(1,now-start)/1e6,2),'MB/s',flush=True);last[0]=now
        with client.open_sftp() as sftp:
            try:offset=sftp.stat(remote+'/input.zip').st_size
            except FileNotFoundError:offset=0
            if offset>output.stat().st_size:raise ValueError('Remote archive larger than source')
            print('RESUME_UPLOAD',offset,flush=True)
            with output.open('rb') as local,sftp.open(remote+'/input.zip','ab') as target:
                target.set_pipelined(True)
                local.seek(offset)
                while True:
                    chunk=local.read(1024*1024)
                    if not chunk:break
                    target.write(chunk);offset+=len(chunk);progress(offset,output.stat().st_size)
            if sftp.stat(remote+'/input.zip').st_size!=output.stat().st_size:raise ValueError('Incomplete upload')
        command='/home/star/agentic_world/.venv/bin/python -m zipfile -e '+remote+'/input.zip '+remote
        _,stdout,stderr=client.exec_command(command,timeout=600)
        if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode())
        print('STAGED',remote,flush=True)
    finally:client.close()


if __name__=='__main__':main()
