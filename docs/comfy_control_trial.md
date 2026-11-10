# Native ComfyUI H3 ControlNet trial

Date: 2026-10-03. This is a separate backend experiment, not a replacement
of the validated full-INT8 DiffSynth renderer.

## Data clarification

The original `GTA_0906 (2).zip` has source code and controller/ingestor SQLite
databases, but no world-state JSONL or finalized Parquet. Rechecked the original
ZIP directory and ingestor schema. `accepted_records` stores four deduplication
keys, not entity payloads.

`video_to_whitebox.zip` DOES contain inferred world scene payloads. For example:
`test_data/GTA_fps5/segment_00001/scene.json` has version 2, 13 geometry proxies,
two character tracks, camera keyframes and nine explicit assumptions. The file
states that dimensions, depth and camera motion are visually approximated.
Use these payloads with INFERRED provenance. "No world payload in either ZIP"
would be incorrect without the qualifier "raw recorded GTA world snapshots."

The new `ep_20260812_090725_f8a5.zip` supplies the recorded world snapshots.
It is partial, not a complete static environment. See the episode report.

## Exact implementation

- ComfyUI: commit `95539f56344958339e39b7582a476267d489b0ee`.
- Source: `/home/star/agentic-world-comfy/ComfyUI`.
- Environment: `/home/star/agentic-world-comfy/.venv`.
- Entry point: `scripts/run_comfy_control.py` calls native ComfyUI nodes directly
  without a browser/server: loaders, MiniMaxH3ReferenceToVideo,
  MiniMaxH3FunControlNetApply, MiniMaxH3SigmaShift, KSampler and VAEDecode.
- Python wrapper: `renderer/comfy_h3.py`, class `ComfyH3Renderer`.
- GPU: physical GPU 3, one model stage at a time.
- Reused existing pruned INT8 Ref2VA DiT and INT8 Qwen text encoder.
- Reused original video VAE at
  `/mnt/hdd/MiniMax-H3/Ref2VA/video_vae/source/model.safetensors`.
- Newly downloaded control branch only:
  `/mnt/hdd/agentic-world-models/model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors`.
  Repository Kijai/MiniMax-H3-experimental, revision
  `294f771095af5b03c38a26aab3ad6498b88092b2`, 4,531,220,608 bytes.
  Downloaded via hf-mirror.com after checking existing named cache/model roots.

Checkpoint headers were inspected before download. The existing FULL INT8 base
has adaLN input width 2688, while this control branch has width 8. The existing
PRUNED INT8 base has width 8 and is the selected compatible counterpart. This
experiment therefore does not establish full-INT8-base compatibility.

Source VAE lacks latents_mean/std keys in its safetensors. ComfyUI initializes
these from built-in constants. Runner checks them against source config after
casting to the actual VAE buffer dtype. An initial exact FP32 comparison failed
on expected FP16 rounding; the corrected same-dtype exact check is used.

Torch 2.9.1+cu128 is reused read-only. This ComfyUI version disables its optimized
comfy-kitchen CUDA backend on cu128, so the eager path is used. No upgrade of
the existing H3 environment was performed.

## Input and measured scope

Actual input: episode agent_001/video_00000.mkv. Resample source timestamps from
20 fps to 24 fps using previous-frame hold, without looping or end padding.
Output: 832x480, 73 frames, 24 fps. Seed 43; Euler/simple, cfg=1; video/audio
sigma shifts 12/3. One RGB reference image plus RGB Canny control frames.
The source frame is resized to the output canvas; this is not camera calibration.

This trial's Canny is computed from recorded RGB. It is NOT a world-state
compiler result, semantic-ID conditioning, metric-depth constraint, or an
end-to-end calibrated world-consistency benchmark.

Smoke result: three real denoising steps completed. All ten control blocks
executed three times with finite nonzero measured residuals. Output decoded as
73 distinct frames. The first smoke's total runtime was about 80 seconds.
Source/condition/output and a contact sheet are under `outputs/comfy_control/smoke3`.

Forty-step controlled/baseline results are written under
`outputs/comfy_control/compare40`. `metadata.json` records calls, timings and
pixel checks; `evaluation.json` records decoded media checks and diagnostic
edge agreement. These metrics are not semantic or authoritative-world accuracy.

Both real runs completed, each with 73 decoded distinct frames at 24 fps:

| Measurement | Control strength 1 | Baseline strength 0 |
| --- | ---: | ---: |
| Sampling + decode/save seconds | 281.11 | 222.68 |
| Control calls per injection layer | 40 | 0 |
| Symmetric edge distance, pixels | 3.915 | 19.092 |
| Edge F1 within 3 pixels | 0.8574 | 0.4577 |

Same seed, reference, prompt, 40 steps and base checkpoint. Total process time
including loading and cleanup was 563.93 seconds. GPU memory was observed around
34 GB during controlled sampling, not instrumented as an exact peak.
Contact-sheet review shows the controlled sequence follows the source's garage
door change and camera composition, whereas the baseline largely retains the
initial open-door composition. HUD/minimap edges also leak into the controlled
result: mask recording UI before producing future conditions.

After generation exited, the existing GroundingDINO/SAM2.1/DINOv2 pipeline
successfully processed generated frame 36 (1.5 seconds). It produced nine
observations, nine masks and a (9, 384) embedding array, including a person
confidence 0.904 and door confidence 0.844. The remaining lower-confidence
cabinet/bottle proposals are not manually validated object facts. No observation
was inserted into authoritative world memory.

Twenty-five regression/renderer contract tests passed. All GPU jobs exited and
GPU memory returned to the idle baseline. Output media, JSON reports and
comparison.jpg were also retrieved to local `agentic_world/outputs/comfy_control/compare40`.

## Reproduce

From `/home/star/agentic_world`:

```bash
CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com \
  /home/star/agentic-world-comfy/.venv/bin/python scripts/run_comfy_control.py \
  --comfy /home/star/agentic-world-comfy/ComfyUI \
  --source input_data/ep_20260812_090725_f8a5/agents/agent_001/video_00000.mkv \
  --output outputs/comfy_control/new_compare --steps 40 --compare

.venv/bin/python scripts/evaluate_comfy_control.py outputs/comfy_control/new_compare
.venv/bin/python -m unittest scripts.test_comfy_renderer
```

Use a fresh output directory per run. CLI also accepts `--control-video` for
externally compiled RGB control frames and `--reference-image`; external depth/
layout preprocessing contracts must be verified separately. It never interprets
arbitrary integer semantic IDs as trained conditions.

```python
from renderer.comfy_h3 import ComfyH3Renderer

path = ComfyH3Renderer().render(
    prompt="Scene description referring to <Picture 1>.",
    source_video="source.mp4",
    output_dir="outputs/new_control_rollout",
    # control_video="compiled_rgb_control.mp4",
    # first_frame="reference.png",
)
```

No ComfyUI web service is required for this headless backend. The existing
prototype CLI still defaults to the unchanged DiffSynth backend. Integration
of calibrated world-derived conditions is a subsequent step, not claimed here.
