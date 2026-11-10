"""Restore the complete no-proxy source package from tar.zst shards."""
from pathlib import Path
import json, os, sys, tarfile, time
import zstandard as zstd

src=Path('/mnt/e/GTA_555_upload_no_proxy/shards')
dst=Path('/mnt/e/GTA_555_complete_rebuild_20261025/source')
dst.mkdir(parents=True,exist_ok=True)
shards=sorted(src.glob('*.tar.zst'))
status=dst.parent/'EXTRACTION_STATUS.json'
start=time.time()
for index, path in enumerate(shards,1):
    marker=dst.parent/f'.{path.stem}.done'
    if marker.exists(): continue
    before=time.time(); files=0; bytes_out=0
    with path.open('rb') as fh, zstd.ZstdDecompressor().stream_reader(fh) as reader:
        with tarfile.open(fileobj=reader,mode='r|') as archive:
            for member in archive:
                # tar paths are produced by our own assembler; reject traversal anyway.
                target=(dst/member.name).resolve()
                if not str(target).startswith(str(dst.resolve())+os.sep): raise RuntimeError(member.name)
                archive.extract(member,dst)
                files+=1; bytes_out+=member.size
                if files%1000==0:
                    status.write_text(json.dumps({'status':'extracting','shard':index,'total_shards':len(shards),'files_in_shard':files,'bytes_restored':bytes_out,'elapsed_sec':time.time()-start},indent=2))
    marker.write_text(json.dumps({'shard':path.name,'files':files,'bytes':bytes_out,'seconds':time.time()-before}))
    status.write_text(json.dumps({'status':'extracting','shard':index,'total_shards':len(shards),'completed':index,'elapsed_sec':time.time()-start},indent=2))
status.write_text(json.dumps({'status':'EXTRACTED','shards':len(shards),'elapsed_sec':time.time()-start},indent=2))
print(status.read_text(),flush=True)
