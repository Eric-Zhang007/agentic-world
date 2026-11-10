"""Extract small control files from the tail shard, leaving source restore alone."""
from pathlib import Path
from compression import zstd
import tarfile
import json

root=Path('E:/GTA_555_complete_rebuild_20261025')
out=root/'inspection'
out.mkdir(exist_ok=True)
entries=[]
with zstd.open('E:/GTA_555_upload_no_proxy/shards/non_jsonl_0090.tar.zst','rb') as r, tarfile.open(fileobj=r,mode='r|') as t:
    for m in t:
        entries.append({'path':m.name,'size':m.size,'type':m.type.decode()})
        if m.isfile() and m.size<5_000_000 and (m.name.startswith(('tools/','method/source/','method/environment/')) or m.name in ('data/index.json','data/episodes.json','data/segments.json','method/batch_run_manifest.json')):
            p=out/m.name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(t.extractfile(m).read())
            print(m.name,flush=True)
(out/'tail_inventory.json').write_text(json.dumps(entries,indent=2))
print('FINISHED',len(entries),flush=True)
