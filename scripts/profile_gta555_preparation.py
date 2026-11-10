"""Measure real source preparation without changing accepted outputs."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import shutil
from scripts.prepare_gta_spool_passages import work


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-root', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(a.summary.read_text())['source']
    if a.cache_root:
        a.cache_root.mkdir(parents=True, exist_ok=True)
        copied = a.cache_root / Path(source['record_index']).name
        if not copied.exists():
            shutil.copy2(source['record_index'], copied)
        source = {**source, 'record_index': str(copied), 'frame_cache_root': str(a.cache_root / 'frames')}
    profiler = cProfile.Profile()
    result = profiler.runcall(work, (source, str(Path(source['record_index']).parent), str(a.output)))
    profiler.dump_stats(a.output / 'preparation.prof')
    with (a.output / 'profile.txt').open('w') as stream:
        pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(35)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
