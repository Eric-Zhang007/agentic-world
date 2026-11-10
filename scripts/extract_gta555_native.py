"""Native streaming extraction with timed progress and completed-shard resume."""
from compression import zstd
from pathlib import Path
import json
import os
import shutil
import tarfile
import time
import traceback


def main():
    root=Path('E:/GTA_555_complete_rebuild_20261025')
    dst=root/'source'
    dst.mkdir(parents=True,exist_ok=True)
    shards=sorted(Path('E:/GTA_555_upload_no_proxy/shards').glob('*.tar.zst'))
    status=root/'EXTRACTION_STATUS.json'
    started=time.time()
    state={'status':'extracting','pid':os.getpid(),'total_shards':len(shards),'completed_shards':0}
    last=0
    def save(force=False):
        nonlocal last
        if not force and time.time()-last<5:return
        last=time.time()
        state.update(updated_unix=last,elapsed_sec=last-started)
        temp=status.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2),encoding='utf-8')
        os.replace(temp,status)
    try:
        save(True)
        for index,path in enumerate(shards,1):
            marker=root/f'.{path.stem}.done'
            if marker.exists():
                state['completed_shards']+=1
                continue
            state.update(shard=index,shard_name=path.name,files_in_shard=0,bytes_restored=0,compressed_bytes_read=0)
            save(True)
            with path.open('rb') as fh, zstd.ZstdFile(fh,'rb') as reader, tarfile.open(fileobj=reader,mode='r|') as archive:
                for member in archive:
                    state['current_file']=member.name
                    target=dst/member.name
                    if not target.resolve().is_relative_to(dst.resolve()):raise ValueError(member.name)
                    if member.isdir():target.mkdir(parents=True,exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with archive.extractfile(member) as inp, target.open('wb') as out:
                            while chunk:=inp.read(4*1024*1024):
                                out.write(chunk)
                                state['bytes_restored']+=len(chunk)
                                state['compressed_bytes_read']=fh.tell()
                                save()
                    elif member.islnk():
                        link_source=(dst/member.linkname).resolve()
                        if not link_source.is_relative_to(dst.resolve()):raise ValueError(member.linkname)
                        if not link_source.is_file():raise FileNotFoundError(f'Hardlink source missing: {member.linkname}')
                        target.parent.mkdir(parents=True,exist_ok=True)
                        if target.exists():
                            if os.path.samefile(target,link_source):continue
                            target.unlink()
                        try:os.link(link_source,target)
                        except OSError as exc:
                            if getattr(exc,'winerror',None) not in (1,50):raise
                            shutil.copyfile(link_source,target)
                            state['copied_hardlinks']=state.get('copied_hardlinks',0)+1
                    else:raise ValueError(f'Unsupported archive entry: {member.name}, type={member.type!r}')
                    state['files_in_shard']+=1
                    state['compressed_bytes_read']=fh.tell()
                    save()
            marker.write_text(json.dumps({'shard':path.name,'files':state['files_in_shard'],'bytes':state['bytes_restored']}))
            state['completed_shards']+=1
            save(True)
            print('COMPLETED',path.name,flush=True)
        state['status']='EXTRACTED'
        save(True)
    except BaseException as exc:
        state.update(status='FAILED',error=str(exc))
        save(True)
        traceback.print_exc()
        raise


if __name__=='__main__':main()
