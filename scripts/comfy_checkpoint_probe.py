"""Inspect small safetensors headers without loading checkpoint tensors."""
import json
from pathlib import Path
import struct

import requests


def summarize(header):
    return {
        "metadata": header.get("__metadata__"),
        "selected": {k: v for k, v in header.items()
                     if any(x in k for x in (
                         "blocks.0.adaln", "blocks.0.after_proj", "control_proj_in",
                         "control_blocks.9.after_proj"))},
    }


def main():
    result = {"local": {}, "remote": {}}
    for p in Path("/mnt/hdd/MiniMax-H3-Quant").rglob("*.safetensors"):
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        result["local"][str(p)] = summarize(h)
    repo = "Kijai/MiniMax-H3-experimental"
    session = requests.Session()
    meta = session.get(f"https://hf-mirror.com/api/models/{repo}?blobs=true", timeout=60)
    meta.raise_for_status()
    meta = meta.json()
    result["remote_revision"] = meta["sha"]
    for item in meta["siblings"]:
        name = item["rfilename"]
        if "model_patches/" not in name or "2.0" not in name:
            continue
        url = f"https://hf-mirror.com/{repo}/resolve/{meta['sha']}/{name}"
        # Stream and close after the header even if a server ignores Range.
        with session.get(url, headers={"Range": "bytes=0-2097151"}, stream=True, timeout=90) as r:
            r.raise_for_status()
            n = struct.unpack("<Q", r.raw.read(8))[0]
            if n > 32 * 1024**2:
                raise ValueError(f"Unexpected header length {n}")
            header = json.loads(r.raw.read(n))
        result["remote"][name] = {
            "url": url, "size": item.get("size", item.get("lfs", {}).get("size")),
            **summarize(header)}
    root = Path("/home/star/agentic_world/outputs/comfy_control")
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoint_probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
