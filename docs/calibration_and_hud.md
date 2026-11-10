# Recorded-world calibration and HUD exclusion

Verified 2026-10-03 on the existing server. No source archive, recorded world
payload, original H3 environment or model weights were modified.

## Scope and results

Episode `ep_20260812_090725_f8a5`, first ten contiguous MKV segments per agent.
Each segment actually decoded to 80 frames at 20 fps. The source archive lacks
the later recorder's segment CSV and captured screen rays; the available source
code is newer than the episode and is not proof of its exact runtime build.

The original camera FOV, native Euler rotation and viewport sizes are used as
candidate observations. GTA ROT_ZXY basis and vertical-FOV projection were tested
against actual video landmarks, not silently assumed correct.

### Clock alignment

`audit_episode_alignment.py` fits centered affine sidecar->server transforms
using the lowest RTT quartile and soft-L1 loss. Every fourth selected pair is
held out. No fitting on held-out pairs.

| Metric | agent_000 | agent_001 |
| --- | ---: | ---: |
| Clock holdout absolute residual P95 | 3.874 ms | 9.302 ms |
| Fitted clock drift | 1.146 ppm | 15.559 ppm |
| Maximum raw showinfo PTS vs decoded CFR index discrepancy, first800 frames | 24.001 ms | 28.806 ms |

These are receipt-clock fit residuals, NOT capture/exposure synchronization
accuracy. In particular agent_001 low-RTT cutoff is still 108.3 ms. Its full
episode receipt-vs-PTS drift spans about 3.45 seconds, so a single whole-episode
offset would not be justified.

We decode all preceding segments and use cumulative frame count, verified against
raw frame IDs/PTS. The runtime adapter accepts only this verified prefix.
It does not assume `segment_number * 4s` for arbitrary segments, extrapolate
outside the calibration interval, interpolate through >250ms telemetry gaps,
or join entities across dataset ID/lifecycle changes.

### Spatial and residual-time calibration

Image landmarks come from GroundingDINO `person. car.` proposals, with conflicting
labels, tiny (<40px tall), clipped and HUD-overlapping landmarks excluded.
The candidate identity is the single other recorded player, not a free association
to whichever world point minimizes error. Detections remain approximate landmarks;
they are not surveyed points or a manually annotated benchmark.

Camera/position interpolation unwraps Euler angles. It explicitly scales focal
length from the original render viewport to the recorded viewport: agent_001
2560x1600 -> 1280x720 is anisotropic, and must not be treated as a square-pixel
camera at the output resolution.

Fit five limited corrections: residual time offset, focal scale, pitch bias,
yaw bias, and a matching-only entity-center Z anchor. Entire four-second segments
with index modulo3==2 are held out. Model selection uses training error only.
Pass gate: held-out median<20px, P95<50px, no fitted parameter at a bound.

agent_001 passes this narrow test:

- 33 fitting landmarks, 14 held-out landmarks.
- Held-out median: 6.985px before -> 4.163px after.
- Held-out P95 after: 21.233px, measured at 1280x720.
- Residual video->telemetry offset: -0.104134 seconds.
- Focal scale 1.004197, pitch +0.711679 degrees, yaw -0.149461 degrees.
- Matching-anchor Z offset +0.079612 world units; entity position remains unchanged.
- Valid server-time interval: 100.965735 .. 137.465735 seconds.
- Nuisance-refitted training median within1px of optimum admits offsets
  -0.179134 .. -0.054134 seconds. This is a sensitivity range, NOT a confidence
  interval. Do not report millisecond exposure calibration.

Horizontal FOV plus free focal scale can be confounded with vertical FOV. The
effective fitted intrinsics are the practical result, not proof of engine FOV
semantics. No metric depth/bbox extent calibration is claimed.

agent_000 is `insufficient_evidence`: only three usable landmarks remain (one
held out). Earlier rejected fits contained car/cursor false positives. No
camera correction is activated for this stream.

`EpisodeAlignment` exports segment2 into 80 per-frame camera/world snapshots.
Eight sampled original-video frames matched the other player's persistent ID
through the existing Hungarian matcher. Pixel errors were
3.61, 5.49, 6.26, 6.04, 5.74, 11.06, 4.08, 40.12. The last-frame error remains
visible; these are not eight perfect observations. Missing geometry means bbox
IoU and metric depth error remain unavailable.

## HUD exclusion

Two explicit, recording-specific normalized profiles:

- `configs/gta_hud.json`: agent_001 lower-left minimap and upper-right watermark.
- `configs/gta_hud_agent000.json`: also conservatively excludes the top console.

Profiles are manually inspected for this prefix, not universal HUD detection.
Mouse cursors/dynamic overlays elsewhere are not comprehensively removed.

`perception/hud.py` preserves output dimensions and pixel coordinates:

1. Keep raw inputs and persist a binary invalid-pixel mask.
2. Reference image uses OpenCV Telea fill only inside the mask. Filled pixels
   remain invalid, never authoritative evidence.
3. Compute Canny from original RGB, then zero the invalid region and dilated
   boundary. Computing Canny after black rectangles would create false edges.
4. Perception cleans its input, rejects any bbox intersecting invalid pixels,
   clears invalid mask pixels and excludes them from embeddings.
5. Projected world anchors inside declared HUD regions are unobservable, not
   missing objects. Camera intrinsics are not changed by this operation.

The optional profile is exposed by `ComfyH3Renderer.render(hud_profile=...)`,
`run_comfy_control.py --hud-profile`, `probe_perception.py --hud-profile`, and
`PerceptionPipeline` config key `hud_profile`.

Actual same-seed40-step H3 trial generated73 frames at832x480/24fps:

- HUD excluded fraction: 5.579%.
- Raw and decoded Canny nonzero pixels inside invalid mask: zero.
- Reference pixel change outside mask: zero.
- Source HUD-edge retention within2px: 17.61% before -> 0% after.
- Valid-region symmetric edge distance: 2.977px before -> 3.069px after.
- Visual review: minimap removed; smooth/inpainted lower-left texture remains.
  Zero edge retention is not OCR proof of every possible HUD absence.
- Cleaned generated video successfully passed GroundingDINO/SAM2.1/DINOv2.
  No generated observation or fill was inserted into the world database.

This HUD trial uses the same original segment0 as the previous control experiment
for comparison. Segment0 is NOT the segment2 used for validated world projections.
The default prototype CLI is unchanged; calibrated source alignment and HUD-clean
rendering are separate verified components, not a claimed complete calibrated
world->geometry-control->generation evaluation.

## Artifacts and commands

Server:

```text
/home/star/agentic_world/outputs/calibration/audit.json
/home/star/agentic_world/outputs/calibration/projection_evidence.json
/home/star/agentic_world/outputs/calibration/projection_calibration.json
/home/star/agentic_world/outputs/calibration/aligned_segment2/timeline.json
/home/star/agentic_world/outputs/comfy_control/hud_clean40/
```

Commands from `/home/star/agentic_world`:

```bash
.venv/bin/python scripts/audit_episode_alignment.py \
  input_data/ep_20260812_090725_f8a5 outputs/calibration
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/collect_projection_evidence.py \
  input_data/ep_20260812_090725_f8a5 outputs/calibration configs/gta_hud.json
.venv/bin/python scripts/fit_episode_projection.py outputs/calibration
.venv/bin/python scripts/export_calibrated_segment.py \
  input_data/ep_20260812_090725_f8a5 outputs/calibration \
  outputs/calibration/aligned_segment2 --hud configs/gta_hud.json
CUDA_VISIBLE_DEVICES=3 /home/star/agentic-world-comfy/.venv/bin/python \
  scripts/run_comfy_control.py --comfy /home/star/agentic-world-comfy/ComfyUI \
  --source input_data/ep_20260812_090725_f8a5/agents/agent_001/video_00000.mkv \
  --output outputs/comfy_control/new_hud_trial --steps 40 \
  --hud-profile configs/gta_hud.json
```

Thirty-two tests passed, including camera axes, nonuniform resize, clock holdout,
HUD boundary exclusion, unchanged world position, gap rejection and existing
renderer/matching regressions. No new model/dependency download was required.
