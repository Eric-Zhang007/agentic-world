# Local Fresh Neural Inference Readiness

Checked 2026-11-09. Local Windows and Ubuntu WSL only; no remote host, API,
download, installation, model forward pass, or cached prediction replay.
Only this document was edited. Weight presence below means nonempty files,
not verified model deserialization. No inference latency was measured.

## Verdict

Subsequent verified progress: the new `scripts/infer_local_depth.py` adapter ran
three genuine Depth Anything V2-small forwards using the native source/checkpoint
listed below, in the existing CUDA WSL environment. No installation or download
was needed. See `docs/local_tracking_and_depth.md` and
`outputs/local_fresh_depth_20261109/report.json`. The original two server scripts
remain blocked unchanged; this is a relative-depth component, not VGGT/SAM
end-to-end initialization.

**Neither existing script is runnable unchanged for fresh offline inference.**
CUDA works in WSL, and useful native checkpoints exist, but the server deployment
described elsewhere in this repository is not the local deployment. A partial
depth/embedding path may be possible with existing assets and explicit local
source imports; it has not been validated and is not a working whitebox pipeline.

## Live Environment Results

GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB total, driver 546.30.
Total memory is not free memory; no VRAM-fit claim is made.

| Interpreter | Python | Torch / CUDA result |
| --- | --- | --- |
| WSL `/home/zjc/.venvs/cross-fact-gpu/bin/python` | 3.10.20 | 2.0.1+cu118; build CUDA 11.8; `cuda.is_available()` true |
| WSL `/usr/bin/python3` | 3.12.3 | 2.10.0+cpu; CUDA unavailable |
| WSL `/home/zjc/.venvs/fact-run/bin/python` | 3.12.3 | 2.10.0+cpu; CUDA unavailable |
| WSL `/home/zjc/analysis_venv/bin/python` | 3.12.3 | torch absent |
| WSL `/home/zjc/embodied-failure-dataset/.venv/bin/python` | 3.10.20 | torch absent |
| WSL `/home/zjc/hermes-agent/venv/bin/python` | 3.11.15 | torch absent |
| WSL `/home/zjc/efbench-merge-8c898e549ad342948d661f13f0fb84a6/.venv/bin/python` | 3.11.15 | torch absent |
| WSL `/mnt/d/fact-stack/venvs/fact-b16/bin/python` | 3.12.3 | torch absent; cv2 present |
| Windows `C:/Python314/python.exe` | 3.14.3 | torch absent |
| Windows `C:/Users/zjc/AppData/Local/Programs/Python/Python312/python.exe` | 3.12.10 | torch absent |

Transformers was absent in all completed probes. VGGT, SAM2, standalone SAM,
and DINOv2 were not import-discoverable in those probes. The CUDA WSL environment
also has discoverable torchvision, huggingface_hub, safetensors, cv2, PIL, numpy;
discovery does not prove every extension imports successfully.

`D:/lewm/le-wm/.venv/Scripts/python.exe -B -` did not complete its import probe
and was stopped by its identified PID. On-disk package metadata shows torch
2.5.1+cu121, torchvision 0.20.1+cu121, transformers 5.6.2 and OpenCV 4.13.0.92;
current imports/CUDA are NOT confirmed for that environment.
Other discovered but unprobed environments include WSL `.frida-venv` and
`D:/fact-stack/venvs/fact-assets`; this is not an exhaustive environment census.

## Verified Local Files

Paths below are Windows paths unless explicitly marked WSL; `D:/` is `/mnt/d/`
inside WSL.

| Model | File | Bytes |
| --- | --- | ---: |
| GroundingDINO | `D:/fact-stack/models/groundingdino/groundingdino_swint_ogc.pth` | 693997677 |
| SAM2 tiny | `D:/fact-stack/models/sam2/sam2_hiera_tiny.pt` | 155906050 |
| SAM3 | `D:/fact-stack/models/sam3/checkpoints/sam3.pt` | 3450062241 |
| Depth Anything V2 base | `D:/fact-stack/models/depth-anything-v2/checkpoints/depth_anything_v2_vitb.pth` | 389961218 |
| Depth Anything V2 small | `D:/fact-stack/models/depth-anything-v2/checkpoints/depth_anything_v2_vits.pth` | 99218434 |
| Metric3D | `D:/fact-stack/models/Metric3D/weight/metric_depth_vit_small_800k.pth` | 150120967 |
| DINOv2 small | `C:/Users/zjc/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` | 88283115 |
| DINOv2 giant | `C:/Users/zjc/.cache/torch/hub/checkpoints/dinov2_vitg14_pretrain.pth` | 4546108579 |

WSL also has SAM3 `sam3.pt` (3450062241 bytes) and `config.json` under
`/home/zjc/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/`.
The old statement that SAM3 is zero bytes is stale. SAM3 is not a SAM2 drop-in.

Native source directories exist at `D:/fact-stack/repos/{GroundingDINO,sam2,sam3,Depth-Anything-V2,Metric3D,dinov3}`.
DINOv2 source exists at `C:/Users/zjc/.cache/torch/hub/facebookresearch_dinov2_main`.
Their dependencies and checkpoint compatibility were not validated.
The named `depth-anything-v2-small-hf` directory yielded no files in the inspected
model tree. Windows Map Anything HF cache contains only `refs/main` (40 bytes),
not model weights. No VGGT snapshot or vendor source was found in the inspected
workspace/default HF roots. Absence claims are limited to those roots.

## Script-Specific Blockers

`scripts/probe_perception.py` performs genuine GroundingDINO -> SAM2 -> DINOv2
forwards, but hardcodes `/mnt/hdd/agentic-world-models/models.json`. That root and
`/home/star` are absent locally. Its default image/output also reference that
server. It requires Transformers SAM2 classes and compatible HF snapshots;
the native `.pth`/`.pt` files above are not those snapshots. It skips SAM2 and
DINOv2 when detection is empty, so a successful empty report alone does not
validate all three networks. This script provides no depth.

`scripts/reconstruct_vggt.py` requires CUDA, missing `vendor/vggt` source and a
`facebook/VGGT-1B` HF snapshot containing `config.json` and `model.safetensors`.
It defaults to physical GPU 2; this laptop requires GPU 0. Use both offline
environment settings and `--local-files-only`: the script otherwise invokes
snapshot resolution against its hardcoded mirror. It produces predicted-scale
depth, not calibrated metric depth. One image is accepted by its CLI, but
model execution and memory fit remain untested.

## Reproducible Commands

PowerShell inventory (read-only):

```powershell
wsl --list --quiet
wsl -d Ubuntu -- nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
Get-ChildItem D:/fact-stack/models -Recurse -Depth 4 -File | Select-Object FullName,Length
Get-ChildItem $HOME/.cache/torch/hub/checkpoints | Select-Object Name,Length
Get-ChildItem $HOME/.cache/huggingface/hub/models--facebook--map-anything -Recurse -Depth 3
```

Bounded WSL import probe, using the exact confirmed GPU interpreter:

```powershell
@'
import sys, importlib.util
print(sys.executable, sys.version)
for name in ['torch','torchvision','transformers','huggingface_hub','safetensors','cv2','PIL','numpy','vggt','sam2','dinov2']:
    print(name, importlib.util.find_spec(name))
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
'@ | wsl -d Ubuntu -- timeout 45 /home/zjc/.venvs/cross-fact-gpu/bin/python -B -
```

Future invocation shapes, **blocked, not executed**, from the WSL workspace
`/mnt/c/Users/zjc/omgd/agentic_world`. They write outputs and therefore were
not run during this read-only assessment. Replace IMAGE and OUT explicitly:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/zjc/.venvs/cross-fact-gpu/bin/python scripts/reconstruct_vggt.py \
  IMAGE --output OUT.npz --gpu 0 --local-files-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/zjc/.venvs/cross-fact-gpu/bin/python scripts/probe_perception.py \
  --image IMAGE --output OUT --device cuda:0 --assert-detections
```

Next engineering work would require either compatible local HF snapshots and
dependency availability plus a configurable manifest path, or explicit adapters
for the native models. Existing depth/DINO assets justify investigating a
partial offline route, not claiming an end-to-end fresh neural reconstruction.
