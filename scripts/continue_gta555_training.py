"""Continue deterministic mining/export after full source preparation.

Exports remain review candidates. This does not mark visual QA complete or
replace the previously delivered training archive.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ffmpeg', required=True)
    p.add_argument('--prepared-source', type=Path,
                   help='Process a frozen subset of completed segments without waiting for missing inputs')
    p.add_argument('--reviewed-root', type=Path, help='Reuse reviews only after exact media comparison')
    p.add_argument('--exclude-target-index', type=Path, help='Search only targets absent from this earlier point index')
    a = p.parse_args()
    if a.prepared_source and not (a.root / 'snapshot.json').exists():
        subprocess.run([sys.executable, '-m', 'scripts.snapshot_gta555_passages',
                        '--prepared', str(a.prepared_source), '--output', str(a.root)],
                       cwd=Path(__file__).resolve().parents[1], check=True)
    else:
        a.root.mkdir(parents=True, exist_ok=True)
    if a.prepared_source and not (a.root / 'trajectory_points.jsonl').is_file():
        raise ValueError('Snapshot is incomplete: trajectory_points.jsonl is missing')
    status = a.root / 'TRAINING_PIPELINE_STATUS.json'

    def save(stage, phase, **details):
        value = {'stage': stage, 'status': phase, 'updated_at': time.time(), **details}
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        for attempt in range(30):
            try:
                temporary.replace(status)
                break
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(.1)

    while not a.prepared_source:
        try:
            prep = json.loads((a.root / 'PREPARATION_STATUS.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            prep = {}
        if prep.get('status') == 'prepared':
            break
        save('source_preparation', 'waiting', completed=prep.get('completed', 0),
             total=prep.get('total'), failures=prep.get('failed', 0))
        time.sleep(20)
    jobs = [
        ('survey', 'scripts.survey_gta_passage_pairs', ['--index', str(a.root / 'trajectory_points.jsonl'), '--output', str(a.root)]),
        ('match', 'scripts.match_gta_passage_pairs', ['--root', str(a.root)]),
        ('export', 'scripts.export_gta_passage_training', ['--root', str(a.root), '--release', str(a.prepared_source or a.root), '--workers', '2']),
        ('review_sheets', 'scripts.review_gta_passage_targets',
         ['--root', str(a.root), '--batch', 'pending_audit_full_source']),
    ]
    environment = {**os.environ, 'FFMPEG_BINARY': a.ffmpeg}
    if a.exclude_target_index:
        jobs[0][2].extend(['--exclude-target-index', str(a.exclude_target_index)])
    if a.reviewed_root:
        jobs.insert(3, ('reuse_reviews', 'scripts.reuse_gta_passage_reviews',
                       ['--source', str(a.reviewed_root), '--target', str(a.root)]))
    for stage, module, arguments in jobs:
        if stage == 'review_sheets' and (a.root / 'pending_audit_full_source' / 'index.json').exists():
            save(stage, 'complete', note='Preserved existing immutable review batch')
            continue
        command = [sys.executable, '-u', '-m', module, *arguments]
        log_path = a.root / (stage + '.log')
        save(stage, 'running', command=command, log=str(log_path))
        with log_path.open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                       env=environment, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                save(stage, 'running', pid=process.pid, log=str(log_path),
                     log_bytes=log_path.stat().st_size)
                time.sleep(10)
        if process.returncode:
            save(stage, 'failed', exit_code=process.returncode, log=str(log_path))
            raise SystemExit(process.returncode)
        save(stage, 'complete', log=str(log_path))
    save('visual_review_and_mixture', 'pending_review',
         note='Candidate export completed; no claim of final training acceptance or final package.')


if __name__ == '__main__':
    main()
