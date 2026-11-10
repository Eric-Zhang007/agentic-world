"""Decode one representative delivered clip per flagged source for direct review."""
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reaudit_delivery_media import audit


def main():
    base=ROOT/'outputs/dataset_reaudit'
    output=base/'priority_visual'
    output.mkdir(parents=True,exist_ok=True)
    sources=[json.loads(line) for line in (base/'metadata/source_clock_rates.jsonl').read_text().splitlines()]
    selected={r['sample_ids'][0] for r in sources if r['flagged']}
    archive=Path('E:/GTA_555_complete_rebuild_20261025/unified_delivery/GTA_memory_training_unified.zip')
    with zipfile.ZipFile(archive) as bundle:
        for split in ('train','validation','test'):
            for index,line in enumerate(bundle.read(f'manifests/{split}.jsonl').decode().splitlines()):
                entry=json.loads(line)
                sid=entry['sample_id']
                if sid not in selected:
                    continue
                path=output/(sid+'.json')
                if path.exists():
                    continue
                entry.update(key=f'{split}:{index}',split=split)
                result=audit(bundle,entry)
                (output/(sid+'.jpg')).write_bytes(result.pop('_contact_jpeg'))
                path.write_text(json.dumps(result,indent=2))
                print(sid,flush=True)


if __name__=='__main__':
    main()
