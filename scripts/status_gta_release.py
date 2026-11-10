"""Report live data-preparation stages without reading large source records."""
from collections import Counter
import json
from pathlib import Path
import argparse


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,default=Path('D:/GTA_memory_release_20261018'))
    root=parser.parse_args().release
    result={}
    for stage,pattern,field in [('qualification','qualification/*/summary.json','status'),
                                ('visual_review','visual_review/*/decision.json','status'),
                                ('exports','export_verdicts/*.json','status')]:
        counts=Counter()
        for path in root.glob(pattern):
            if '_superseded_' in path.name:continue
            try:counts[json.loads(path.read_text(encoding='utf-8'))[field]]+=1
            except (OSError,ValueError,KeyError):counts['UNREADABLE']+=1
        result[stage]=dict(counts)
    result['index_bytes']={p.name:p.stat().st_size for p in root.glob('*.sqlite3')}
    result['complete']=(root/'DELIVERY.json').is_file()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
