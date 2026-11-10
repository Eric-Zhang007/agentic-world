"""Preserve and retire reviews made before exact sequential frame extraction."""
import json
from pathlib import Path
import shutil


def main():
    root=Path('E:/GTA_memory_release_20261018').resolve()
    windows={w['sample_id']:w for p in (root/'qualification').glob('*/windows.jsonl') for w in map(json.loads,p.read_text().splitlines())}
    retired=[]
    for path in sorted((root/'visual_review').glob('*/decision.json')):
        key=path.parent.name;w=windows[key]
        segment=w['episode_id']+'_'+w['agent_id']+'_'+w['segment']
        complete=root/'review_frames'/segment/'complete.json'
        sheet=path.parent/'rgb_video.jpg'
        if complete.exists() and sheet.exists() and sheet.stat().st_mtime_ns>=complete.stat().st_mtime_ns:
            continue
        for source in (path.parent,root/'dataset/samples'/key,root/'export_verdicts'/(key+'.json')):
            if not source.exists():continue
            destination=root/'superseded_random_seek'/source.relative_to(root)
            if not source.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(root):raise ValueError('Move outside release workspace')
            if destination.exists():raise ValueError('Superseded destination already exists')
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(destination))
        retired.append(key)
    (root/'random_seek_retirement.json').write_text(json.dumps({'reason':'Random-seek frames were not guaranteed to equal source decoder indices','samples_requiring_new_review':retired},indent=2))
    print('Retired',len(retired),'reviews; preserved originals')


if __name__=='__main__':main()
