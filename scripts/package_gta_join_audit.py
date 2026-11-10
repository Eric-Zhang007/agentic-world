"""Package measured join audit and sparse visual decisions, not training labels."""
import json
from pathlib import Path
import zipfile


def main():
    root = Path('E:/GTA_memory_prepared')
    audit = root/'record_join_audit_0921_v1'
    review = root/'record_join_review_0921_v1'
    decisions = Path(__file__).resolve().parents[1]/'configs/gta_join_review_0921_v1.json'
    output = root/'record_join_review_0921_v1.zip'
    if output.exists():
        raise ValueError('Existing archive will not be overwritten')
    labels = json.loads(decisions.read_text(encoding='utf-8'))
    index = json.loads((review/'review_index.json').read_text(encoding='utf-8'))
    if {r['sheet'] for r in index} != {r['sheet'] for r in labels['reviews']}:
        raise ValueError('Review sheet/decision mismatch')
    by_sheet = {r['sheet']: r for r in labels['reviews']}
    for row in index:
        row['visual_review'] = by_sheet[row['sheet']]
        row['review_status'] = 'SPARSE_MANUAL_REVIEW_COMPLETE'
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in ('summary.json', 'audited.jsonl'):
            archive.write(audit/name, 'record_audit/'+name)
        archive.writestr('visual_review/index.json', json.dumps(index, indent=2))
        archive.write(decisions, 'visual_review/decisions.json')
        for row in index:
            archive.write(review/row['sheet'], 'visual_review/'+row['sheet'])
        archive.write(Path(__file__).resolve().parents[1]/'docs/gta_record_join_audit.md', 'README.md')
        archive.writestr('STATUS.txt', 'Audit evidence only, not a finished training dataset. 3377/3379 windows pass actual record joins. Eight sparse visual sheets were reviewed; no full-clip visibility acceptance. The large derived SQLite index and source media remain local.\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity failure')
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size, 'reviewed_sheets': len(index)}))


if __name__ == '__main__':
    main()
