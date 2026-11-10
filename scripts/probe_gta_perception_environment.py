"""Read-only environment inventory; optional single-frame remote inference smoke."""
import argparse
import io
import json
from pathlib import Path
import shlex
import sys


PYTHON = "/home/star/agentic_world/.venv/bin/python"
PROBE = "/home/star/agentic_world/scripts/probe_perception.py"
INVENTORY = r'''
import importlib.metadata as m, json, pathlib, sys
import torch
from transformers import AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
manifest = json.loads(pathlib.Path('/mnt/hdd/agentic-world-models/models.json').read_text())
models = {k: v for k, v in manifest.items() if 'path' in v}
for v in models.values():
    p = pathlib.Path(v['path'])
    v['exists'] = p.is_dir()
    v['weight_bytes'] = sum(f.stat().st_size for f in p.glob('*.safetensors'))
print(json.dumps({'python': sys.executable, 'versions': {n: m.version(n) for n in
    ['torch','torchvision','transformers','huggingface-hub','numpy','Pillow','opencv-python-headless']},
    'cuda_available': torch.cuda.is_available(), 'torch_path': torch.__file__,
    'models': models}, indent=2))
'''


def run(client, command):
    _, stdout, _ = client.exec_command(command, timeout=300)
    stdout.channel.set_combine_stderr(True)
    chunks = []
    while True:
        chunk = stdout.channel.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    output = b"".join(chunks).decode("utf-8", errors="replace")
    if stdout.channel.recv_exit_status():
        raise RuntimeError("Remote probe failed:\n" + output[-6000:])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Read one local video frame; never uploads the video.")
    parser.add_argument("--frame", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--text", default="person. car. truck. bus. motorcycle. bicycle.")
    parser.add_argument("--assert-detections", action="store_true")
    args = parser.parse_args()
    if args.frame < 0 or not 0 <= args.threshold <= 1:
        parser.error("frame must be nonnegative and threshold must be in [0, 1]")
    payload = None
    if args.video:
        import cv2
        capture = cv2.VideoCapture(str(args.video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Cannot decode requested frame")
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError("Cannot encode decoded frame")
            payload = encoded.tobytes()
        finally:
            capture.release()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from server_exec import connect
    client = connect()
    try:
        prefix = "PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 "
        inventory = json.loads(run(client, prefix + PYTHON + " -c " + shlex.quote(INVENTORY)))
        print(json.dumps({"inventory": inventory}, indent=2), flush=True)
        if payload is None:
            return
        directory = run(client, "mktemp -d /tmp/gta-perception-env-XXXXXXXX").strip()
        remote_image = directory + "/input.png"
        with client.open_sftp() as sftp:
            sftp.putfo(io.BytesIO(payload), remote_image)
        command = [PYTHON, PROBE, "--image", remote_image, "--output", directory + "/result",
                   "--device", "cuda:0", "--text", args.text, "--threshold", str(args.threshold)]
        # Keep the normal empty-frame contract; assert only after reading its report.
        run(client, prefix + shlex.join(command))
        with client.open_sftp() as sftp:
            with sftp.open(directory + "/result/result.json") as stream:
                result = json.load(stream)
        observations = result["observations"]
        print(json.dumps({"smoke": {
            "success": result["success"], "source_video": str(args.video), "frame": args.frame,
            "remote_artifacts": directory, "image_size": result["image_size"],
            "timings": result["timings"], "peak_gpu_mib": result["peak_gpu_mib"],
            "embedding_shape": result["embedding_shape"],
            "objects": [{k: o[k] for k in ("category", "confidence", "mask_area", "mask_predicted_iou")}
                        for o in observations]}}, indent=2), flush=True)
        if args.assert_detections and not observations:
            raise RuntimeError("Smoke assertion failed: no detections; inference itself completed")
    finally:
        client.close()


if __name__ == "__main__":
    main()
