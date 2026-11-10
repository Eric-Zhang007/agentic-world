"""Check local files against the package's segment and spool contracts.

This is a readiness inventory, not a media decode or visual quality audit.
Reports are written outside the immutable source tree.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit(source, output, metadata=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Write audit output outside the source bundle")
    output.mkdir(parents=True, exist_ok=True)
    metadata = Path(metadata) if metadata else source
    index = read_json(metadata / "data/index.json")
    segments = read_json(metadata / "data/segments.json")["segments"]
    episodes = {e["path"]: e for e in index["episodes"]}
    totals = Counter()
    by_episode = {}
    jobs = []
    missing = []

    def present(relative, context, allow_empty=False):
        path = source / relative
        ok = path.is_file() and (allow_empty or path.stat().st_size > 0)
        if not ok:
            missing.append({"path": relative, "context": context})
        return ok

    global_files = ["data/episodes.json", "data/global_index.sqlite3",
                    "audit/files.jsonl", "method/frozen_inputs.zip",
                    "method/original_scripts/inputs.py", "tools/reproduce.py",
                    "method/native/libEGL.so.1", "method/native/libEGL_mesa.so.0",
                    "method/source/src/wm_engine/proxy/cli.py",
                    "data/worlds/gta5_scene_v3_20260908/manifest.json"]
    globals_present = {p: present(p, "global") for p in global_files}
    spool_status = {}
    if globals_present["data/episodes.json"]:
        for e in read_json(source / "data/episodes.json"):
            spool_status[e["episode_id"]] = {}
            for kind, spec in e["spools"].items():
                spool_status[e["episode_id"]][kind] = {
                    k: present(v, e["episode_id"] + "/" + kind)
                    for k, v in spec.items()}
    world_manifest = source / global_files[-1]
    world_status = {}
    if world_manifest.is_file():
        world_status = {p: present("data/worlds/gta5_scene_v3_20260908/" + p, "world", allow_empty=True)
                        for p in read_json(world_manifest).get("sha256", {})}

    for s in segments:
        e = episodes[s["episode_relative"]]
        eid = e["episode_id"]
        folder = "data/" + s["output_relative"]
        agent, segment = Path(s["output_relative"]).parts[-2:]
        context = eid + "/" + agent + "/" + segment
        checks = {"rgb": present("data/" + s["rgb_relative"], context)}
        for name in ("compiler_manifest.json", "validation.json", "frames.jsonl"):
            checks[name] = present(folder + "/" + name, context)
        counts = by_episode.setdefault(eid, Counter())
        counts["segments"] += 1
        totals["segments"] += 1
        for key, value in checks.items():
            counts[key] += int(value)
            totals[key] += int(value)
        spools = spool_status.get(eid, {})
        required_spools = [spools.get(k, {}) for k in ("world", agent)]
        render_globals = {k: v for k, v in globals_present.items() if k != 'audit/files.jsonl'}
        manifest_present = present('data/' + e['manifest'], context)
        ready = (all(checks.values()) and all(render_globals.values()) and manifest_present
                 and bool(world_status) and all(world_status.values())
                 and all(bool(v) and all(v.values()) for v in required_spools))
        totals["ready"] += int(ready)
        jobs.append({"episode": eid, "agent": agent, "segment": segment,
                     "rgb_relative": "data/" + s["rgb_relative"],
                     "expected_frames": s["frames"], "usable_frames": s["usable_frames"],
                     "files": checks, "ready": ready})
    extraction_path = source.parent / "EXTRACTION_STATUS.json"
    extraction = read_json(extraction_path) if extraction_path.exists() else None
    report = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "source": str(source), "metadata": str(metadata),
              "scope": "Nonempty file presence; no media decode or visual validation",
              "extraction_snapshot": extraction, "totals": dict(totals),
              "global_files": globals_present, "world_files": world_status,
              "original_assembly_inventory_available": globals_present['audit/files.jsonl'],
              "spools": spool_status, "by_episode": by_episode,
              "missing_references": len(missing)}
    for name, value in (("readiness.json", report), ("render_jobs.json", jobs)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    with (output / "missing_files.jsonl").open("w", encoding="utf-8") as stream:
        for row in missing:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps({"totals": dict(totals), "missing_references": len(missing),
                      "report": str(output / "readiness.json")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    audit(args.source, args.output, args.metadata)
