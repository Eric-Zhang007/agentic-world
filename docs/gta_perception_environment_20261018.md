# GTA perception environment readiness - 2026-10-18

## Decision

Use the existing remote GroundingDINO tiny -> SAM2.1 tiny -> DINOv2 small
stack through `C:/Users/zjc/omgd/server_exec.py`. Actual GTA inference passed
offline on GPU 0. No installation, weight download, training, H3 modification,
raw-data modification, or main-pipeline edit was performed. This establishes
runtime readiness, not visibility accuracy or bulk throughput qualification.

## Verified remote runtime

- Python: `/home/star/agentic_world/.venv/bin/python`.
- Existing entrypoint: `/home/star/agentic_world/scripts/probe_perception.py`.
- Transport: existing SSH helper through local SOCKS port 10085.
- Four RTX 4090 devices reported 49140 MiB each; all idle at initial inspection.
  Only GPU 0 was used. Availability is a point-in-time observation.
- Packages: torch metadata 2.9.1, torchvision 0.24.1, transformers 5.18.0,
  huggingface-hub 1.33.0, NumPy 2.4.6, Pillow 12.3.0,
  opencv-python-headless 5.0.0.93. CUDA and model imports passed.
- Torch is imported read-only from the existing H3 site-packages. Probe commands
  set `PYTHONDONTWRITEBYTECODE=1`; no H3 environment activation or package changes.
- No missing dependency for the tested remote path. This is an SSH-invoked CLI,
  not a verified persistent detector HTTP service. Listening-port inspection did
  not establish any perception API.

Manifest: `/mnt/hdd/agentic-world-models/models.json`. All three paths and
nonempty safetensors were checked and actually loaded:

```text
/mnt/hdd/agentic-world-models/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71
/mnt/hdd/agentic-world-models/hub/models--facebook--sam2.1-hiera-tiny/snapshots/de431c4043854a71d8101e17995dfe596bf101a5
/mnt/hdd/agentic-world-models/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056
```

Weight sizes respectively: 689359096, 155908064, 88249960 bytes. No network
model resolution is needed. If later provisioning becomes necessary, the
existing setup script supports `HF_ENDPOINT=https://hf-mirror.com`; do not run
downloads for this already-complete stack.

## Exact Windows commands

Run from `C:/Users/zjc/omgd`. Default `python` is `C:/Python314/python.exe`.
The new probe imports the parent workspace's existing SSH helper, reads one
local video frame, uploads only its PNG to a unique remote `/tmp` directory,
and runs the existing remote inference script. It leaves inspectable temporary
artifacts and prints their paths; no local output directory is created.

```powershell
python agentic_world/scripts/probe_gta_perception_environment.py
python agentic_world/scripts/probe_gta_perception_environment.py --video E:/GTA_555/0916/agents/agent_000/video_00000.mkv --frame 120 --assert-detections
python agentic_world/scripts/probe_gta_perception_environment.py --video E:/GTA_555/0916/agents/agent_001/video_00000.mkv --frame 120 --assert-detections
python agentic_world/scripts/probe_gta_perception_environment.py --video E:/GTA_555/0916/agents/agent_000/video_00000.mkv --frame 120 --threshold 1
```

Default proposal text: `person. car. truck. bus. motorcycle. bicycle.`
Use `--text` to change it. Frame numbers are zero-based decoder positions,
not synchronized source timestamps or a validated paired-data join.

To repeat inference on the already-staged first frame, this exact command is
available while its temporary directory remains present:

```powershell
python server_exec.py "PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /home/star/agentic_world/.venv/bin/python /home/star/agentic_world/scripts/probe_perception.py --image /tmp/gta-perception-env-xF0yVZXL/input.png --output /tmp/gta-perception-env-xF0yVZXL/repeat --device cuda:0 --text 'person. car. truck. bus. motorcycle. bicycle.'"
```

## Actual smoke results

All inputs were 1280x720, frame 120 of the named first video segment. All three
commands exited zero. Timings include stage model loading, exclude transfer
and local decode, and are single-run measurements, not bulk benchmarks.

| Input | Boxes / nonempty masks | Embeddings | Detector / SAM / embedding seconds |
| --- | --- | --- | --- |
| agent_000 | 5 / 5 | (5,384) | 6.794 / 1.405 / 1.258 |
| agent_001 | 4 / 4 | (4,384) | 1.769 / 0.317 / 0.252 |
| agent_000, threshold 1 | 0 / 0 | (0,384) | 1.794 / 0 / 0 |

Peak PyTorch allocated memory: 1955.48 MiB per process, not total device use.
Artifacts are in `/tmp/gta-perception-env-xF0yVZXL`,
`/tmp/gta-perception-env-jTQovf68`, and `/tmp/gta-perception-env-PfiRHSBw`
respectively, each with `result/result.json`, input, overlay, and embeddings;
positive runs also contain masks and crops. Temporary paths are not durable.
Positive masks ranged from 75 to 19060 pixels. Labels and confidence values
are model proposals, not ground truth; small/ambiguous detections need filtering.

## Local fallback inventory

- Windows GPU: RTX 3060 Laptop, 6144 MiB, driver 546.30; initially 5311 MiB free.
- `C:/Python314/python.exe`: OpenCV, Pillow, NumPy, Paramiko available;
  torch, transformers, ultralytics, sam2, segment_anything absent.
- `D:/lewm/le-wm/.venv/Scripts/python.exe`: torch 2.5.1+cu121,
  transformers 5.6.2, OpenCV available; CUDA and GPU name verified by import.
  Standalone groundingdino and sam2 absent. No local model inference tested;
  do not treat this as a validated replacement for the remote HF stack.
- WSL Ubuntu `/mnt/d/fact-stack/venvs/fact-b16/bin/python`: OpenCV present;
  torch, transformers, groundingdino, sam2 absent.
- `D:/annotation-pipeline/.venv` and `D:/embodied-failure-dataset/.venv`:
  OpenCV present, torch/transformers/sam2/ultralytics absent.
- `D:/EgoVerse/.venv`: torch discoverable; transformers/OpenCV/sam2/ultralytics
  absent. Discovery only, not CUDA inference validation.
- Existing native-format weights:
  `D:/fact-stack/models/groundingdino/groundingdino_swint_ogc.pth` (693997677 bytes),
  `D:/fact-stack/models/sam2/sam2_hiera_tiny.pt` (155906050 bytes).
  These are not drop-in HF snapshot paths for the existing remote entrypoint.
- `D:/fact-stack/models/sam3/checkpoints/sam3.pt` is **zero bytes**, despite
  an older manifest saying already-present. It is not usable.
- `C:/Users/zjc/.cache/huggingface/hub/models--facebook--map-anything`
  contains only a ref in the inspected tree, not downloaded weights.
- Other existing caches: `.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`,
  `dinov2_vitg14_pretrain.pth`, and `.cache/clip/ViT-L-14-336px.pt` under the user
  home. These are not object detector checkpoints.
- No YOLO `.pt`/`.onnx` checkpoint found in checked workspace/cache and
  `D:/fact-stack`, `D:/fact-runs`, `D:/GTA_dataset` locations. Search was not an
  exhaustive scan of every disk/user environment; no CPU YOLO path validated.

## Static geometry quick search

Old project docs `input_data/GTA_0906/docs/proxy_source_geometry.md` and
`proxy_filters.md` reference `D:/GTA_dataset/worlds/gta5_scene_v3_20260908`.
That directory is absent at inspection. A bounded remote directory search
(maximum depth 5) under `/home/star/agentic_world`, `/mnt/hdd`, and
`/home/star/worlds` found no match; missing roots were ignored. This does not
prove absence elsewhere. No geometry was downloaded or reconstructed.

## Handoff boundary

The existing CLI supplies pixel bboxes, scores, full-resolution masks, mask
area, and embeddings for candidate-window frames now. For bulk use, the main
agent should own a persistent model worker, sampling, source joins, staging,
retry policy, HUD handling, and output indexing. The smoke CLI reloads models
per frame and is not an efficient all-window batch runner. GPU 0 is verified,
not reserved. No other GPU was exercised.

No-detection is a valid empty observation, not proof of invisibility. This probe
does not establish occlusion, identity continuity, paired timestamp alignment,
HUD exclusion, or visibility recall. Current tests used no HUD profile.
The source dataset indexing and any API-based review remain outside this work.
