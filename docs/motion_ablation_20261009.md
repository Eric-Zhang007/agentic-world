# Whitebox motion detail versus endpoint control

This is a read-only experiment on the accepted courtyard action, not a new
authoritative world revision. It tests whether one coarse lateral step, staff
sweep and recovery can produce richer animation without changing its events.

## Fixed settings

- Existing native Comfy H3 Ref2VA INT8 + Fun Union 2.0; existing weights/env.
- 1024x768, 73 frames at 24fps, 40 Euler/simple steps, seeds 43 and 71.
- Same original generated appearance image, RGB first-frame guide and prompt.
- Same clean Godot depth sequence, strength 0.85; EasyCache disabled.
- Accepted 75-tick proxy plus initialized state resampled to 73 frames with
  both endpoints preserved. The manifest records all source indices/times.
- Generated outputs are observations only. No world DB is opened for writing.

## Pilot and input correction

The original `motion_ablation_20261009` Canny pilot was stopped after its first
completed video visibly copied preview wire/grid lines into the image. Its
video, perception, review and interrupted logs are retained, not overwritten.
No conclusion about sparse control is drawn from this incomplete pilot.

The preview is now separate from a Godot `--render-pass depth`: camera-space
relative grayscale, near white / far black, no ground grid, lighting, shadows
or proxy colors. The shader maps camera-z 2..18 to white..black; this is a
renderer condition, not a measured monocular depth.

The preview camera also made actors much smaller than in the appearance image.
`calibrate_motion_camera.py` fits an experimental camera to first guided-frame
actor detections. Geometry, poses, action and timing are unchanged; the camera
and all renders live in `outputs/motion_calibrated_20261009`, never in the
authoritative DB. Absolute center-x and top/bottom residuals are 1.7..25.2px.
Width is excluded because armor and held props change the silhouette. This
does NOT calibrate background structures, exact grip, weapon angle or metric
scale. The initial staff pose still differs from the reference and limits
attribution. Corrected matrix: `outputs/motion_depth_ablation_20261009`.

## Temporal control implementation

The native `start_percent`/`end_percent` are denoising intervals, NOT video time.
The experiment uses a worker-local hook, leaving installed Comfy files intact:
unselected control pixels are blacked out before VAE encoding, unselected
latent hints are zeroed, and direct ControlNet residuals are restricted to the
selected target-video rows. Text, reference, guide and audio rows get no direct
control residual in ALL three experimental arms, including the dense arm.
The legacy worker path is unchanged when `temporal_keyframes` is absent.

H3's pixel spans per temporal token are `(1,4,4,4,4)` repeated. Windows are
expanded to entire tokens. Dense controls all 73 frames; events selects
frames 0,34,49,72 and controls 10/73 frames; endpoints selects 0,72 and controls
5/73 frames. Exact spans are recorded in generation metadata.

This is an experimental inference intervention, NOT a trained sparse-control
capability. Temporal VAE compression and self-attention still spread influence.
Zeroing residuals does not guarantee that the unconstrained interval is
independent, or that the selected states will be obeyed. Simplified actor
silhouettes, partial scene geometry and reference mismatch remain limitations.

No terminal RGB guide is used: a raw whitebox guide would also constrain the
appearance. Native guide support and actual reliable endpoint adherence are
separate questions. This experiment tests geometric conditions only.

## Run on the server

```bash
cd /home/star/agentic_world
.venv/bin/python scripts/prepare_motion_ablation.py \
  --attempt outputs/motion_calibrated_20261009 \
  --output outputs/motion_ablation_NEW \
  --reference /home/star/agentic_world/outputs/h3_scene_20261008/frame_011.png \
  --depth-dir outputs/motion_calibrated_20261009/depth
.venv/bin/python scripts/run_motion_ablation.py outputs/motion_ablation_NEW --gpus 3,2,1
.venv/bin/python scripts/evaluate_motion_ablation.py outputs/motion_ablation_NEW --all --device cuda:0
.venv/bin/python scripts/summarize_motion_ablation.py outputs/motion_ablation_NEW
```

The evaluator may run while generation proceeds; it waits for each case to be
published in metadata before reading its video. This run explicitly assigns
generation to GPUs 3,2,1 and perception to GPU0. The runner defaults to one
GPU (3); it does not auto-occupy every device. No weights are downloaded.

## Evidence and interpretation

Each case has its original MP4, actual gated control MP4, four keyframe
GroundingDINO/SAM2/DINOv2 results, deterministic role/projection matching,
expected/observed overlays, endpoint displacement diagnostics and a visual
review blinded to the condition label. Review sees 25 uniformly sampled frames
plus keyframes. It is not frame-perfect contact verification. Approximate
proxy boxes and uncertain detector identities are not metric 3D measurements.

`summary.json`, `report.md` and two synchronized four-panel videos are created
only after all six real cases have generation and evaluation artifacts.
The four panels are whitebox, dense, events and endpoints. Two seeds are a
small diagnostic comparison, not a measured production success rate.

## Observed result

All six corrected depth cases generated successfully. Sampling took
303.4..333.7 seconds per clip. Existing weights, environments and native Comfy
source were not modified. GPU memory returned to idle after generation and
perception completed. 89 core/regression tests and 8 engine contract tests
passed; compileall passed. Both preview and new depth rendering were actually
executed on all 75 action frames plus the initialized state.

All six boundary-contract reviews returned REGENERATE: natural motion and
appearance persisted, but the intended displaced final state did not. One
endpoint-only seed was judged to show a single horizontal sweep; the other
five were judged as flourishes, partial rotations or holds. No case passed
both action and endpoint criteria. These are fallible agent visual judgments,
supported by inspectable samples, not an estimate of H3's general capability.

The first review prompt overemphasized exact intermediate proxy facing/grips.
Those reviews are retained as `strict_proxy_review.json`. The final
`visual_review.json` has contract_version=2, explicitly permits natural torso
turns, weight transfer and temporary grip adjustment between critical events,
and judges action semantics and final state. Initial grip mismatch is recorded
as uncertainty. This correction did not make any clip pass endpoint review.

Detector boxes sometimes include the staff or miss the opponent. Their raw
displacement errors are diagnostics, not reliable calibrated performance
numbers. Do not rank the control methods by these errors or by subjective
richness scores. Use the synchronized videos and per-case all-frame sheets.

`input_verification.json` confirms the experimental camera changed neither
entity data nor action poses/timing, and the persistent DB still matches its
committed snapshot: 15 entities, revision 2, two historical actions, t=5.625s.
No generated deviation was silently adopted as world truth.

The next discriminating experiment should first make initial pose/grip and
appearance conditions mutually consistent, then test a single translation
with distinct terminal placement. A visually consistent terminal RGB guide
and a stronger event-specific weapon condition can then be evaluated
separately. Current sparse residual gating is experimental and should not be
treated as native trained endpoint control or promoted to the default renderer.
