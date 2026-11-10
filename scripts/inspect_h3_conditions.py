"""Read-only H3 audit: stdlib, no torch/imported model code, no weight downloads.

Local: python -B agentic_world/scripts/inspect_h3_conditions.py \
    --server-exec C:/Users/zjc/omgd/server_exec.py --metadata
Remote: python3 -B inspect_h3_conditions.py --metadata
JSON goes to stdout only. Remote mode sends this source in memory, not via SFTP.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.request


REPO = Path("/mnt/hdd/DiffSynth-Studio")
DEPLOY = Path("/home/star/minimax-h3/deploy")
MODEL = Path("/mnt/hdd/MiniMax-H3")
QUANT = Path("/mnt/hdd/MiniMax-H3-Quant")
LIMIT = 2 * 1024 * 1024
SKIP = {".git", "__pycache__", ".venv", "site-packages", "node_modules",
        ".Trash-1000", "lost+found"}
PATTERN = re.compile(
    r"controlnet|control_rows|control_hints|control_layers|control_in_dim|"
    r"control_apply_audio|FL2VA/|ref_visual_pos_parts|cond_rows_count|"
    r"frames\[-1\]|convert\(\"RGB\"\)|control_video|semantic", re.I
)


def read_small(path):
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError("metadata/source exceeds 2 MiB")
    return data


def source_record(path):
    try:
        data = read_small(path)
        text = data.decode("utf-8")
        return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                "matches": [{"line": n, "text": line}
                            for n, line in enumerate(text.splitlines(), 1)
                            if PATTERN.search(line)]}
    except (OSError, ValueError) as exc:
        return {"path": str(path), "error": str(exc)}


def inventory(root):
    result = {"root": str(root), "exists": root.exists(), "components": {}}
    if not root.exists():
        return result
    for part in ("FL2VA", "Ref2VA"):
        for component in ("transformer", "text_encoder", "video_vae", "audio_vae"):
            directory = root / part / component
            files = sorted(directory.rglob("*.safetensors"))
            sizes = {str(p.relative_to(directory)): p.stat().st_size for p in files}
            entry = {"files": len(files), "bytes": sum(sizes.values())}
            index = directory / "model.safetensors.index.json"
            if index.exists():
                info = json.loads(read_small(index))
                expected = sorted(set(info["weight_map"].values()))
                entry["index_shards"] = len(expected)
                entry["missing_or_empty"] = [
                    name for name in expected if sizes.get(name, 0) == 0]
                entry["index_tensor_bytes"] = info.get("metadata", {}).get("total_size")
            result["components"][part + "/" + component] = entry
    return result


def scan_roots(roots):
    results = []
    for root in roots:
        result = {"root": str(root), "exists": root.exists(), "entries": 0,
                  "control_matches": [], "symlink_directories_not_followed": [],
                  "errors": [], "truncated": False}
        if root.exists():
            for directory, dirs, files in os.walk(
                    root, followlinks=False,
                    onerror=lambda exc: result["errors"].append(str(exc))):
                dirs[:] = sorted(d for d in dirs if d not in SKIP)
                for name in dirs + sorted(files):
                    path = Path(directory) / name
                    result["entries"] += 1
                    if path.is_symlink() and name in dirs:
                        result["symlink_directories_not_followed"].append(str(path))
                    # Inspect names and stat only, including hashed cache blobs.
                    if re.search(r"control[-_]?net", str(path), re.I):
                        if name in dirs or path.suffix.lower() in {
                                ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
                                ".incomplete"} or "/blobs/" in str(path):
                            try:
                                result["control_matches"].append({
                                    "path": str(path), "directory": name in dirs,
                                    "bytes": path.stat().st_size if name in files else None})
                            except OSError as exc:
                                result["errors"].append(str(exc))
                if result["entries"] > 300000:
                    result["truncated"] = True
                    break
        results.append(result)
    return results


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect refused: metadata requests must not fetch blobs")


def fetch_metadata(url, kind):
    result = {"url": url}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "H3-read-only-audit/1"})
        with urllib.request.build_opener(NoRedirect).open(request, timeout=25) as response:
            if int(response.headers.get("Content-Length", "0")) > LIMIT:
                raise ValueError("metadata response exceeds 2 MiB")
            raw = response.read(LIMIT + 1)
            if len(raw) > LIMIT:
                raise ValueError("metadata response exceeds 2 MiB")
            result.update(status=response.status, response_bytes=len(raw),
                          sha256=hashlib.sha256(raw).hexdigest())
        if kind == "hf":
            obj = json.loads(raw)
            result.update(repo=obj.get("id"), revision=obj.get("sha"),
                          weights=[s for s in obj.get("siblings", [])
                                   if s["rfilename"].endswith(".safetensors")])
        elif kind == "ms":
            obj = json.loads(raw)
            if not obj.get("Success"):
                raise ValueError("ModelScope metadata reports failure")
            result["weights"] = [s for s in obj["Data"]["Files"]
                                 if s["Path"].endswith(".safetensors")]
        else:
            text = raw.decode("utf-8")
            result["relevant_lines"] = [
                line for line in text.splitlines()
                if re.search(r"condition|blocks|config|Layout|semantic|Depth|Canny|legacy",
                             line, re.I) and "<video" not in line]
    except (OSError, ValueError, KeyError) as exc:
        result["error"] = str(exc)
    return result


def audit(metadata):
    sources = [
        DEPLOY / "README.md", DEPLOY / "h3_quant_single.py",
        REPO / "diffsynth/pipelines/minimax_h3_audio_video.py",
        REPO / "diffsynth/models/minimax_h3_controlnet.py",
        REPO / "diffsynth/models/minimax_h3_dit_comfy.py",
        REPO / "diffsynth/configs/model_configs.py",
        REPO / "diffsynth/utils/state_dict_converters/minimax_h3_controlnet.py",
        REPO / "examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Fun-Controlnet-Union.py",
    ]
    before = [source_record(p) for p in sources]
    roots = [MODEL, QUANT, Path("/mnt/hdd/h3-models"),
             Path("/mnt/hdd/agentic-world-models"), REPO / "models",
             Path.home() / ".cache/huggingface",
             Path.home() / ".cache/modelscope", Path("/home/star/minimax-h3/run")]
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "MODELSCOPE_CACHE"):
        value = os.environ.get(name)
        if value and Path(value) not in roots:
            roots.append(Path(value))
    result = {
        "time": datetime.now(timezone.utc).isoformat(),
        "local_time": datetime.now().astimezone().isoformat(),
        "safety": "stdlib only; no GPU API; no model import; no filesystem writes",
        "sources": before, "inventory": inventory(MODEL),
        "quant_weights": [{"path": str(p), "bytes": p.stat().st_size}
                          for p in sorted(QUANT.rglob("*.safetensors"))],
        "scan": scan_roots(roots), "scan_excluded_directory_names": sorted(SKIP),
    }
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=10)
    result["repo_commit"] = git.stdout.strip() or git.stderr.strip()
    result["host_memory"] = [
        line for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemTotal:", "MemAvailable:", "SwapFree:"))]
    stat = os.statvfs("/mnt/hdd")
    result["disk_available_bytes"] = stat.f_bavail * stat.f_frsize
    if metadata:
        records = []
        for repo in ("PAI/MiniMax-H3-Fun-Controlnet-Union",
                     "alibaba-pai/MiniMax-H3-Fun-Controlnet-Union",
                     "alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0"):
            record = fetch_metadata("https://hf-mirror.com/api/models/" + repo +
                                    "?blobs=true", "hf")
            records.append(record)
            revision = record.get("revision", "")
            if re.fullmatch(r"[0-9a-f]{40}", revision):
                records.append(fetch_metadata(
                    "https://hf-mirror.com/" + repo + "/raw/" + revision +
                    "/README.md", "readme"))
        records.append(fetch_metadata(
            "https://modelscope.cn/api/v1/models/PAI/"
            "MiniMax-H3-Fun-Controlnet-Union/repo/files?Revision=master&Recursive=true",
            "ms"))
        result["remote_metadata"] = records
    after = [source_record(p) for p in sources]
    result["source_hashes_stable_during_audit"] = all(
        a.get("sha256") and a.get("sha256") == b.get("sha256")
        for a, b in zip(before, after))
    print(json.dumps(result, ensure_ascii=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-exec", type=Path, help="Existing local SSH helper")
    parser.add_argument("--metadata", action="store_true",
                        help="Fetch bounded model metadata/cards, never weight bodies")
    args = parser.parse_args()
    if args.server_exec:
        payload = base64.b64encode(Path(__file__).read_bytes()).decode("ascii")
        code = "import base64;exec(compile(base64.b64decode(%r),%r,'exec'))" % (
            payload, "inspect_h3_conditions.py")
        command = "python3 -B -c " + shlex.quote(code)
        if args.metadata:
            command += " --metadata"
        return subprocess.call(
            [sys.executable, "-B", str(args.server_exec.resolve()), command],
            cwd=str(args.server_exec.resolve().parent))
    audit(args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
