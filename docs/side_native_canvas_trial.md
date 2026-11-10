# Native depth canvas diagnostic

This isolated side-session experiment does not accept a rollout or mutate a
world database. It tests one confirmed preprocessing discrepancy.

## Confirmed input discrepancy

The saved `live/segment_00` condition renderer used 960x640, then resized its
depth images to 1024x768. RGB guidance was already 1024x768. The scaling factors
were 1.0667 horizontally and 1.2 vertically. This changes projected proportions.
Frame alignment checks found a true boundary frame followed by 123 simulation
ticks at 24 Hz; no simple one-frame shift was found.

## Change

`run_video_reconcile_trial.render_proxy` accepts explicit canvas dimensions.
`conditions` now renders depth, including any historical prefix, directly at
1024x768. `read_depth_frames` rejects wrong-sized images instead of stretching
them. Preview defaults remain unchanged for existing historical workflows.

The diagnostic uses `replay_native_depth_canvas.py` and writes only to
`outputs/identity_repair_20261013/side_native_canvas`. Compared with
`stable_anchor_depth1/ablation.json`, its actual H3 command changes only the
control input path and output directory. World geometry, camera parameters,
prompt, RGB reference, seed 102, 40 steps, strength 1.0 and disabled cache stay
the same. It also creates a native 4:3 preview for visual review.

Server tests: 24 tests passed across native canvas, video reconciliation,
reference ablation and saved-prompt suites. Runtime checks confirmed native
1024x768 PNGs, a uint8 array of shape (124,768,1024,3), pixel equality between
the depth boundary and its packed frame, and unchanged reference pixels.

## Remaining alignment limitations

The separate real-agent boundary audit reports `usable_for_strong_depth=false`:
the player's front-versus-side/back orientation, tiger volume/facing, staff
projection and dominant architectural masses still disagree with the accepted
RGB boundary. This experiment fixes canvas distortion, not those errors.

`evaluate_motion_ablation.project` deliberately uses the old source aspect 1.5
to evaluate historical stretched renders. Its current numerical projection
metrics MUST NOT be used to certify native 4:3 output. This trial instead uses
the newly rendered native preview with `evaluate_camera_revisit`. Historical
metrics and accepted world history are not recomputed or overwritten.

The old generation gate also explicitly relaxed source-image weapon angle
agreement. A future alignment gate should distinguish coarse anatomy freedom
from conflicting object placement, facing, long-prop projection and occlusion.
Changing that gate is not part of this isolated canvas ablation.

## Reproduction

From `/home/star/agentic_world`:

```bash
.venv/bin/python scripts/replay_native_depth_canvas.py \
  --record outputs/identity_repair_20261013/stable_anchor_depth1/ablation.json \
  --plan outputs/identity_repair_20261013/live/segment_00/plan \
  --output outputs/identity_repair_20261013/side_native_canvas \
  --gpu 0
```

The script refuses an existing output directory. Choose a new output directory
for a repeated experiment. Generation and review results are recorded below
after the real run completes.

## Actual result

H3 completed successfully, producing 124 frames at 24 fps. The native runner
reported 766.97 seconds including loading and generation. The real-agent camera
audit in `review/review.json` returned FAIL: camera execution, identity and
static world consistency were all CONTRADICTED. Manual inspection of the middle
and final frames also shows the original armored player and an additional
thin staff-bearing humanoid simultaneously. The original player remains visible
at the end, unlike the previous stretched-canvas result, but duplication persists.

The canvas correction is verified; it is not sufficient to fix this example.
This one-seed comparison cannot establish the dominant cause or measure a
general quality improvement. Remaining reference/proxy geometry conflicts are
visible and should be calibrated in a separate experiment, without rewriting
accepted history. Neither candidate was committed.

Artifacts (also downloaded to the matching local output directory):

- `generated.mp4`: corrected-canvas generated video.
- `before_after.mp4`: previous stretched depth versus native depth, same seed.
- `planned_comparison.mp4`: native 4:3 proxy versus generated video.
- `boundary_audit.json`: remaining initial-condition mismatches.
- `review.json`: local copy of the server camera audit.

No new packages or model weights were installed. The changed shared helper is
`scripts/run_video_reconcile_trial.py`; new diagnostic/test entry points are
`scripts/replay_native_depth_canvas.py` and `scripts/test_native_depth_canvas.py`.
