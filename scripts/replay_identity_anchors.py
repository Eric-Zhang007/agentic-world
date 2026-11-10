"""Change only the identity-text suffix of a saved prompt; no world writes."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from state.db import WorldDB
from whitebox.appearance import AppearanceMemory
from agent.render_contract import identity_text

HEADER='Stable appearance descriptions from accepted video memory.'


def main(a):
    source=json.loads((a.compilation/'source.json').read_text())
    if Path(source['world']).resolve()!=a.world.resolve():
        raise ValueError('Compilation belongs to another world')
    prompt=(a.compilation/'render_prompt.txt').read_text(encoding='utf-8')
    if prompt.count(HEADER)!=1:raise ValueError('Expected one recorded identity suffix')
    prefix=prompt[:prompt.index(HEADER)]
    bundle=json.loads((a.compilation/'appearance_memory/request.json').read_text())
    contract=json.loads((a.compilation/'render_contract.json').read_text())
    db=WorldDB(':memory:')
    try:
        with sqlite3.connect(a.world.resolve().as_uri()+'?mode=ro',uri=True) as original:
            original.backup(db.connection)
        if db.get_meta('whitebox_revision')!=bundle['world_revision']:
            raise ValueError('Source world revision changed')
        stored={r['id']:r for r in db.get_meta(AppearanceMemory.META_KEY,[])}
        if any(r['id'] not in stored or r['prompt']!=stored[r['id']]['prompt'] for r in bundle['refs']):
            raise ValueError('Recorded reference provenance changed')
        memory=AppearanceMemory(db)
        identities=contract['rendered_appearance_object_ids']
        bundle['identity_anchors']=[memory.identity_anchor(identity) for identity in identities]
        if any(r is None for r in bundle['identity_anchors']):raise ValueError('Missing identity anchor')
        suffix=identity_text(bundle,identities)
        if '<Picture' in suffix:raise ValueError('Identity text cannot add renderer image tags')
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/'render_prompt.txt').write_text(prefix+suffix,encoding='utf-8')
        (a.output/'experiment.json').write_text(json.dumps({
            'source_compilation':str(a.compilation.resolve()),'world_commit':False,
            'prompt_prefix_unchanged':True,'selected_object_ids_unchanged':identities,
            'identity_anchors':bundle['identity_anchors'],
            'changed':'Object-bound stable anchor suffix instead of crop-budget-dependent descriptions',
            'limitations':'Historical text anchors do not establish unobserved views or resolve real appearance changes.'
        },indent=2),encoding='utf-8')
        print('Identity suffix replay prepared; world unchanged')
    finally:db.connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['world','compilation','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
