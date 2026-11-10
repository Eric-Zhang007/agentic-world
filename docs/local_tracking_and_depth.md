# Local tracking and fresh depth evidence

## Tracking, not repeated manual annotations

`perception.motion_evidence.track_points` reuses existing Lucas-Kanade optical
flow with forward/backward consistency. It preserves point ordering and returns
NaN/false for invalid identities. Tracks outside the image are rejected. Existing
whole-frame motion evidence now uses the same implementation.

`scripts/track_scene_landmarks.py` transfers explicit initial anchors through
sparse image files. `scripts/compare_anchor_sampling.py` compares dense tracking
and direct first-to-last tracking on the SAME decoded video. A lost identity is
not silently reacquired. It can optionally pass surviving static landmarks into
the existing camera-translation diagnostic, producing an uncommitted candidate.

Actual November 9 observations:

| Case | Frames | Dense decode + tracking | Dense surviving | Endpoint surviving |
| --- | --- | --- | --- | --- |
| Urban street | 120 | 1.619 s | 3/3 | 2/3 |
| Rural street | 120 | 1.421 s | 3/3 | 3/3 |

The urban sign-post anchor fails the direct endpoint check but survives sequential
tracking. Rural endpoint/dense results differ by 0.03-0.26 pixels. Survival and
agreement are not ground truth; the marked final images were visually inspected.
Urban base annotations are approximate and roughly 3-7 pixels away from the
independent later manual annotations. Do not turn these into metric measurements.

`outputs/local_anchor_tracking_20261109/urban_camera_update/` reruns the same
urban video, then fits translation against fixed world anchors. Residuals are
2.44, 1.91, 3.23 pixels. All entities remain unchanged; actors are deliberately
NOT maintained by this camera-only experiment. An actual Godot render was also
run. This does not accept the frame into world history.

The video is 120 frames at 24 fps: final relative PTS is 4.9583 seconds. This is
encoded video time, not the original capture nanosecond timeline. First decoded
RGB vs annotated source RGB mean absolute differences were 0.845 (urban) and
0.890 (rural), on the 0-255 pixel scale, consistent with the selected compressed
clip boundaries. No whitebox GT or dataset camera was used to estimate motion.

Example (PowerShell; use a fresh output directory):

```powershell
python scripts/compare_anchor_sampling.py --video E:/GTA_555_complete_rebuild_20261025/available_source_training/training/samples/ep_20260918_132805_7503_agent_000_segment_00000_f000480/target_rgb.mp4 --annotations configs/urban_source_landmarks.json --scene outputs/local_landmark_fit_20261109/urban/scene.json --output outputs/new_tracking
```

## Fresh neural inference now works for relative depth

Existing local native Depth Anything V2-small source and checkpoint were reused.
Nothing was installed, downloaded or sent to an API. The existing WSL
`/home/zjc/.venvs/cross-fact-gpu/bin/python` successfully loaded the checkpoint and
ran fresh inference on urban, rural and garage images. Predictions are not cached
VGGT outputs. This script does not expose a compatible VGGT reconstruction NPZ.

`outputs/local_fresh_depth_20261109/report.json` records:

- RTX 3060 Laptop GPU, torch 2.0.1+cu118, input size 518.
- Imports/model loading and device transfer: 6.435 s.
- Urban first forward including preprocessing/output transfer: 0.720 s.
- Rural subsequent forward: 0.115 s; garage: 0.107 s.
- Peak torch allocated memory: approximately 446-447 MiB. This is NOT total GPU
  process residency or whole-machine free VRAM.
- Output arrays retain source size 720 x 1280, with finite nonconstant values.

Raw `relative_depth.npy`, color previews and source/depth comparisons are saved.
All three comparisons were inspected. Major foreground/background ordering is
visually plausible, including the garage opening. Thin structures, occluded
surfaces and exact distance remain uncertain. Do not backproject these arbitrary
relative values as metres or inject them into the world without calibration.

Reproduction from project root in WSL:

```bash
/home/zjc/.venvs/cross-fact-gpu/bin/python -B scripts/infer_local_depth.py \
  --repo /mnt/d/fact-stack/repos/Depth-Anything-V2 \
  --checkpoint /mnt/d/fact-stack/models/depth-anything-v2/checkpoints/depth_anything_v2_vits.pth \
  --image outputs/gta_pi_loop_20261102/urban/input.png \
  --output outputs/new_fresh_depth --gpu 0
```

The script supports repeated `--image`, loads one model once, then releases it
on exit. Existing H3 environments and accepted world databases are untouched.

## Remaining integration

Fresh relative depth is now a practical local evidence source. Scale calibration,
mask discovery/segmentation and multi-view alignment are still needed before
automatic 3D primitive extraction. Tracking can reduce repeated annotation, but
cannot identify newly visible objects or infer contact/ownership. These are
components for the agent workflow, not proof of complete autonomous maintenance.
