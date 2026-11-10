"""Provision isolated perception packages, or forward the local SOCKS proxy over SSH."""
import argparse
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading


ROOT = Path("/home/star/agentic_world")
CACHE = Path("/mnt/hdd/agentic-world-models")
MODELS = {
    "detector": "IDEA-Research/grounding-dino-tiny",
    "segmenter": "facebook/sam2.1-hiera-tiny",
    "embedding": "facebook/dinov2-small",
}


def tunnel():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from server_exec import connect

    client = connect()
    transport = client.get_transport()
    remote_port = int(os.environ.get("REMOTE_PROXY_PORT", "10086"))
    local_port = int(os.environ.get("LOCAL_PROXY_PORT", "7897"))
    transport.request_port_forward("127.0.0.1", remote_port)
    print(f"Remote SOCKS: socks5h://127.0.0.1:{remote_port}", flush=True)

    def relay(channel):
        with socket.create_connection(("127.0.0.1", local_port)) as sock:
            try:
                while True:
                    ready, _, _ = select.select([sock, channel], [], [], 60)
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (channel if source is sock else sock).sendall(data)
            finally:
                channel.close()

    try:
        while transport.is_active():
            channel = transport.accept(1)
            if channel:
                threading.Thread(target=relay, args=(channel,), daemon=True).start()
    finally:
        client.close()


def setup():
    CACHE.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UV_CACHE_DIR=str(CACHE / "uv"))
    uv = "/home/star/.local/bin/uv"
    python = ROOT / ".venv/bin/python"
    if not python.exists():
        subprocess.run([uv, "venv", "--system-site-packages", "--python",
                        "/home/star/minimax-h3/.venv/bin/python",
                        str(ROOT / ".venv")], check=True, env=env)
    site = ROOT / ".venv/lib/python3.11/site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (site / "h3_readonly.pth").write_text(
        "/home/star/minimax-h3/.venv/lib/python3.11/site-packages\n")
    subprocess.run([uv, "pip", "install", "--python", str(python),
                    "--no-deps", "opencv-python-headless==5.0.0.93",
                    "PySocks==1.7.1", "socksio==1.0.0"], check=True, env=env)
    print(python)


def download():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    status = {}
    for repo in ("facebook/sam3", "facebook/dinov3-vits16-pretrain-lvd1689m"):
        try:
            info = api.model_info(repo)
            status[repo] = {"gated": info.gated, "revision": info.sha}
        except Exception as exc:
            status[repo] = {"error": str(exc)}
    for name, repo in MODELS.items():
        info = api.model_info(repo)
        path = snapshot_download(
            repo, revision=info.sha, cache_dir=str(CACHE / "hub"),
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
        status[name] = {"repo": repo, "revision": info.sha, "path": path}
        (CACHE / "models.json").write_text(json.dumps(status, indent=2))
        print(name, path, flush=True)
    (CACHE / "models.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["setup", "download", "tunnel"])
    args = parser.parse_args()
    globals()[args.action]()
