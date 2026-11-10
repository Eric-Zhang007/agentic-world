"""Download only the missing compatible control branch through HF mirror."""
import json
from pathlib import Path
import subprocess


def main():
    probe = json.loads(Path("/home/star/agentic_world/outputs/comfy_control/checkpoint_probe.json").read_text())
    name = "model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
    item = probe["remote"][name]
    target = Path("/mnt/hdd/agentic-world-models") / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != item["size"]:
            raise RuntimeError("Existing complete checkpoint has unexpected size")
        print(f"Reusing {target}", flush=True)
        return
    # Check the named cache locations before creating another large checkpoint.
    for root in (Path("/home/star/.cache/huggingface"), Path("/mnt/hdd/MiniMax-H3-Quant")):
        for cached in root.rglob(target.name):
            if cached.stat().st_size == item["size"]:
                target.symlink_to(cached.resolve())
                print(f"Reusing {cached}", flush=True)
                return
    partial = target.with_suffix(".partial")
    subprocess.run([
        "curl", "--fail", "--location", "--continue-at", "-", "--retry", "3",
        "--connect-timeout", "30", "--max-time", "1800", "--speed-time", "120",
        "--speed-limit", "10000", "--output", str(partial),
        item["url"] + "?download=true",
    ], check=True)
    if partial.stat().st_size != item["size"]:
        raise RuntimeError("Downloaded checkpoint length differs from repository metadata")
    partial.replace(target)
    print(json.dumps({"path": str(target), "size": target.stat().st_size,
                      "revision": probe["remote_revision"]}), flush=True)


if __name__ == "__main__":
    main()
