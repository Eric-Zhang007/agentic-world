"""Install ComfyUI dependencies into an isolated overlay; never mutate H3."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/home/star/agentic-world-comfy")
SOURCE = ROOT / "ComfyUI"
H3 = Path("/home/star/minimax-h3/.venv")
UV = "/home/star/.local/bin/uv"
COMMIT = "95539f56344958339e39b7582a476267d489b0ee"


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def main():
    actual = run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
                 capture_output=True).stdout.strip()
    if actual != COMMIT:
        raise RuntimeError(f"Unexpected checkout: {actual}")
    env = ROOT / ".venv"
    if not (env / "bin/python").exists():
        run([UV, "venv", "--python", str(H3 / "bin/python"), str(env)])
    site = env / "lib/python3.11/site-packages"
    (site / "h3_readonly.pth").write_text(
        str(H3 / "lib/python3.11/site-packages") + "\n")
    # Constrain the shared ABI instead of allowing the resolver to upgrade torch.
    constraints = ROOT / "comfy_env_constraints.txt"
    constraints.write_text("torch==2.9.1\ntorchvision==0.24.1\ntorchaudio==2.9.1\n")
    environment = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com",
                       UV_HTTP_TIMEOUT="120")
    run([UV, "pip", "install", "--python", str(env / "bin/python"),
         "--no-deps", "pip"], env=environment)
    run([str(env / "bin/python"), "-m", "pip", "install",
         "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple",
         "--extra-index-url", "https://pypi.org/simple",
         "--constraint", str(constraints), "-r", str(SOURCE / "requirements.txt"),
         "opencv-python-headless==5.0.0.93"],
        env=environment)
    inventory = run([str(env / "bin/python"), "-m", "pip", "list",
                     "--format", "json"], capture_output=True).stdout
    (ROOT / "comfy_env_packages.json").write_text(inventory)
    (ROOT / "comfy_env_deployment.json").write_text(json.dumps({
        "commit": actual, "source": str(SOURCE), "environment": str(env),
        "shared_readonly_site": str(H3 / "lib/python3.11/site-packages"),
        "hf_endpoint": environment["HF_ENDPOINT"],
        "models_downloaded": False, "gpu_inference_run": False,
    }, indent=2))
    run([str(env / "bin/python"), str(SOURCE / "main.py"), "--help"],
        cwd=SOURCE, env=environment)


if __name__ == "__main__":
    main()
