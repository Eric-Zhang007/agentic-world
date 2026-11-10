"""Rebuild condition text and reference IDs after the conditioning contract audit."""
import argparse
import json
from pathlib import Path
from memory.gta_conditioning import reference_prompt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();changed=[]
    for path in (args.release/'dataset/samples').glob('*/sample.json'):
        sample=json.loads(path.read_text());prompt=path.parent/sample['conditions']['prompt']
        expected=reference_prompt(sample['object_references'])
        needs_change=prompt.read_text(encoding='utf-8')!=expected
        provenance='deterministic object-reference mapping; no target RGB caption'
        if sample['conditions'].get('prompt_provenance')!=provenance:needs_change=True
        sample['conditions']['prompt_provenance']=provenance
        for i,pair in enumerate(sample['conditions']['reference_pairs']):
            ref_id=sample['sample_id']+f':ref{i}'
            if pair.get('reference_id')!=ref_id:needs_change=True
            pair['reference_id']=ref_id
        if needs_change:
            prompt.write_text(expected,encoding='utf-8')
            path.write_text(json.dumps(sample,indent=2),encoding='utf-8')
            changed.append(sample['sample_id'])
    log=args.release/'conditioning_repairs.json'
    previous=json.loads(log.read_text()) if log.exists() else []
    log.write_text(json.dumps(sorted(set(previous+changed))),encoding='utf-8')
    print('Conditioning repaired:',len(changed),flush=True)


if __name__=='__main__':main()
