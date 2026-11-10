"""Inspect GTA episode payloads and extract a small, reproducible evidence subset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import zipfile


def inspect(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {"archive": str(archive), "groups": {}, "world_rows": 0}
    with zipfile.ZipFile(archive) as z:
        report["inventory"] = [{"name": e.filename, "bytes": e.file_size}
                               for e in z.infolist() if not e.is_dir()]
        for name in ("manifest.json", "quality_report.json", ".finalize_request.json"):
            report[name] = json.loads(z.read(name))
        ids = {k: set() for k in ("agents", "vehicles", "peds", "objects")}
        maximum = Counter()
        completeness = Counter()
        kinds = Counter()
        sequence_errors = 0
        previous = None
        for line in z.open(".spool/world_state.jsonl"):
            if not line.strip():
                continue
            row = json.loads(line)
            report["world_rows"] += 1
            if report["world_rows"] == 1:
                report["first_world"] = row
            report["last_world"] = row
            seq = row["sample_seq"]
            if previous is not None and seq != previous + 1:
                sequence_errors += 1
            previous = seq
            completeness[row.get("environment", {}).get("world_state_completeness")] += 1
            kinds[row.get("world_state_kind")] += 1
            for group in ids:
                values = row.get(group, [])
                maximum[group] = max(maximum[group], len(values))
                ids[group].update(v["dataset_entity_id"] for v in values)
                if values and group not in report["groups"]:
                    report["groups"][group] = values[0]
        report["unique_entities"] = {k: len(v) for k, v in ids.items()}
        report["max_entities_per_sample"] = dict(maximum)
        report["completeness"] = dict(completeness)
        report["world_state_kind"] = dict(kinds)
        report["sequence_discontinuities"] = sequence_errors
        report["duration_sec"] = (report["last_world"]["server_monotonic_ns"]
                                  - report["first_world"]["server_monotonic_ns"]) / 1e9
        selected = [e.filename for e in z.infolist()
                    if e.filename.endswith(".parquet") or e.filename in (
                        "manifest.json", "quality_report.json", ".finalize_request.json")]
        report["first_client"] = {}
        for agent in ("agent_000", "agent_001"):
            with z.open(f".spool/agents/{agent}/client.jsonl") as f:
                report["first_client"][agent] = json.loads(next(f))
            videos = sorted(e.filename for e in z.infolist()
                            if e.filename.startswith(f"agents/{agent}/video_")
                            and e.filename.endswith(".mkv"))
            report.setdefault("video_segments", {})[agent] = len(videos)
            if videos:
                selected.append(videos[0])
        for name in selected:
            target = (output / name).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError(f"Unsafe archive path: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        report["extracted"] = selected
    (output / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    brief = {k: v for k, v in report.items()
             if k not in ("inventory", "first_world", "last_world", "first_client", "extracted")}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    inspect(a.archive, a.output)
