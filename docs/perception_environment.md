# Perception Environment

## Server

- Host: `star@192.168.0.68`
- SSH transport: local SOCKS `127.0.0.1:10085` via `server_exec.py`
- H3 environment: `/home/star/minimax-h3/.venv` (read-only for this work)
- Perception environment: `/home/star/agentic_world/.venv`
- Model/cache directory: `/mnt/hdd/agentic-world-models`
- Python: 3.11.15
- Shared torch: `2.9.1+cu128`, read-only imported from the H3 environment
- Transformers: `5.18.0` (shared)
- GPUs: 4 x RTX 4090, 48 GiB each

The perception venv was created with `uv venv --system-site-packages` and has an
independent `h3_readonly.pth` pointing at the existing H3 site-packages.
Only `opencv-python-headless`, `PySocks`, and `socksio` are installed in the
perception environment. No H3 files or weights are copied or modified.

## Models

The default public stack is:

- GroundingDINO tiny: `IDEA-Research/grounding-dino-tiny`
- SAM2.1 Hiera tiny: `facebook/sam2.1-hiera-tiny`
- DINOv2 small: `facebook/dinov2-small`

Download metadata and resolved local paths are written to
`/mnt/hdd/agentic-world-models/models.json`. `SAM3` and DINOv3 were checked
first; SAM3 is a gated/manual-access model, so this smoke deployment uses the
public fallback stack.

## Download from hf-mirror.com (preferred)

Checked `/home/star/.cache/huggingface/hub` before downloading: no cached
GroundingDINO/SAM2/DINOv2. Existing DINOv1 at
`/home/star/.cache/vbench/dino_model` was not used or modified.
GroundingDINO's completed new cache was reused. The interrupted SAM2 partial
file was retained; the mirror finished the download in a separate cache blob.

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HOME=/mnt/hdd/agentic-world-models HF_HUB_DISABLE_XET=1 \
/home/star/agentic_world/.venv/bin/python \
/home/star/agentic_world/scripts/setup_perception.py download
```

The script defaults to this mirror. Server direct access succeeded, without
proxy variables. All three downloads completed successfully.

## Optional proxy fallback

The local SOCKS proxy is forwarded to the server by running this on the
development machine:

```powershell
python agentic_world/scripts/setup_perception.py tunnel
```

The server-side downloader then uses:

```bash
HTTPS_PROXY=socks5://127.0.0.1:10086 \
HTTP_PROXY=socks5://127.0.0.1:10086 \
HF_HOME=/mnt/hdd/agentic-world-models \
/home/star/agentic_world/.venv/bin/python \
/home/star/agentic_world/scripts/setup_perception.py download
```

The SSH connection itself still uses the existing `server_exec.py` SOCKS
connection on local port `10085`.

## Real smoke test

After the model download completes:

```bash
HF_HOME=/mnt/hdd/agentic-world-models \
HF_HUB_OFFLINE=1 \
/home/star/agentic_world/.venv/bin/python \
/home/star/agentic_world/scripts/probe_perception.py
```

The script runs sequentially and releases each model before loading the next:
GroundingDINO boxes, SAM2 masks, and DINOv2 crop embeddings. It writes
`result.json`, `overlay.png`, per-object masks/crops, and `embeddings.npy` to
`/home/star/agentic_world/outputs/perception_smoke`.

## Main-process environment

```bash
export PYTHON=/home/star/agentic_world/.venv/bin/python
export HF_HOME=/mnt/hdd/agentic-world-models
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
```

The H3 process must be run separately using its existing documented command.
Do not activate or modify `/home/star/minimax-h3/.venv` for perception.

## Verified results (2026-10-03, server local time)

Both real smoke tests exited zero, with actual local weights and no network:

- H3 example `0.png`: 2 boxes, 2 nonempty masks, embeddings `(2,384)`.
- Existing H3 rollout `int8_conv_full_tav2va.mp4`: middle frame, 832x480,
  7 boxes (1 person, 6 dogs), 7 nonempty masks, normalized embeddings `(7,384)`.
- Video smoke output: `/home/star/agentic_world/outputs/perception_video_smoke`.
- Initial video timings including model loading: detector 1.80s, SAM2 0.31s,
  DINOv2 0.26s. Peak allocated GPU memory: 1998.4 MiB. Single-run measurements.
- Manifest: `/mnt/hdd/agentic-world-models/models.json`.
- Frozen revisions: detector `a2bb814dd30d776dcf7e30523b00659f4f141c71`;
  SAM2 `de431c4043854a71d8101e17995dfe596bf101a5`;
  DINOv2 `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- Runtime versions: torch 2.9.1+cu128, torchvision 0.24.1,
  transformers 5.18.0, huggingface-hub 1.33.0, NumPy 2.4.6,
  Pillow 12.3.0, opencv-python-headless 5.0.0.93,
  PySocks 1.7.1, socksio 1.0.0.
- OpenCV was upgraded only in the perception venv because 4.12's declared
  NumPy upper bound conflicts with the inherited NumPy 2.4.6.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
/home/star/agentic_world/.venv/bin/python \
/home/star/agentic_world/scripts/probe_perception.py \
--video /home/star/minimax-h3/run/int8_conv_full_tav2va.mp4 \
--output /home/star/agentic_world/outputs/perception_video_smoke
```

### Exact Transformers interfaces

- `AutoProcessor` and `AutoModelForZeroShotObjectDetection` for GroundingDINO;
  `post_process_grounded_object_detection(..., threshold=0.25,
  text_threshold=0.25, target_sizes=[(height,width)])` returns boxes in pixels,
  scores and `text_labels` (not `box_threshold` in this version).
- `Sam2Processor`, `Sam2Model`; `input_boxes=[boxes.tolist()]`,
  `multimask_output=False`; `post_process_masks(pred_masks.cpu(),
  original_sizes.cpu())[0][:,0]` yields one original-resolution mask per box.
- `AutoImageProcessor`, `AutoModel` for DINOv2; use masked image crops,
  `last_hidden_state[:,0]`, float32 L2 normalization, 384 dimensions.
- Load all paths from the manifest with `local_files_only=True`; do not
  resolve repo names at runtime. No PYTHONPATH override is necessary.

## Current limitations

The first smoke test uses an explicit text proposal list for GroundingDINO.
SAM2 consumes those boxes, and DINOv2 produces normalized 384-dimensional
embeddings. Video sampling, tracking, depth, and world-state matching remain
owned by the main perception implementation.
The example proxy image produced duplicate boxes and a combined `tree road`
label, so the smoke validates execution, not detection quality. The SAM2 hub
config declares `sam2_video`; loading the image subset with `Sam2Model` emits
a model-type warning but loads all 309 image-model tensors and produces
nonempty masks. Both SAM3 and DINOv3 require manual-access approval.

## Empty-frame contract

Normal inference permits zero detections: exit 0, `success: true`,
`observations: []`, and an empty float32 `embeddings.npy` of shape `(0,384)`.
SAM2 and DINOv2 are not loaded for that frame; both timings are zero. The
input and unannotated overlay are still saved. Consume only the files
referenced by the current result, not masks left from earlier output reuse.

`--assert-detections` is an opt-in smoke-test assertion, not a production
inference requirement. It exits nonzero on empty detections after saving the
valid inference report.

Regression tests on GPU0 only:

- `outputs/perception_positive_test`: real H3 frame, 7 objects, exit 0 with
  `--assert-detections`.
- `outputs/perception_empty_test`: same actual detector with `--threshold 1.0`,
  zero observations, `(0,384)` embeddings, exit 0.
- `outputs/perception_empty_assert_test`: same threshold plus
  `--assert-detections`, expected exit 1.

All paths above are relative to `/home/star/agentic_world`. GPUs 2/3 were not
used by these tests.
