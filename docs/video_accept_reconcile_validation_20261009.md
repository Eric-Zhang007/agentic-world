# Server validation: video acceptance and corrected continuation

Date: 2026-10-09. Project: `/home/star/agentic_world`.
Agent: configured `gpt-6-astra`. Renderer: existing native Comfy H3 Ref2VA INT8
with Fun Union2 depth conditioning, 40 steps, 1024x768, 73 frames at 24 fps,
GPU 3, cache disabled. No new dependencies, weight downloads, or changes to
the installed H3/Comfy environment were required.

The source is an H3-generated fantasy courtyard, not official game footage.
The scene has 15 existing entities and three actors, including an offscreen
simulated NPC. Camera calibration and geometry remain inferred approximations.
Each trial uses a newly initialized DB; it does not rewind the previous world.

## Actual trials

All paths below are under project `outputs/`.

| Trial | Actual outcome | World update |
| --- | --- | --- |
| `video_reconcile_live_v4_20261009` | Native RGB guides resolved to frames 0 and 72. A composited terminal image contaminated the generated background and duplicated the opponent. GPT returned REGENERATE. | Revision 0, no accepted history |
| `video_reconcile_live_depth_20261009` | Clean first RGB guide plus full depth. GPT saw stance adjustment rather than the requested 0.30m sidestep and returned REGENERATE. | Revision 0, no accepted history |
| `video_reconcile_guard_v2_20261009/segment_00` | GPT accepted a three-second nonattacking guard stance, corrected the tiger's predicted root motion, and marked unseen NPC motion as simulated. | Committed revision 1 at approximately 3 seconds |
| `video_reconcile_guard_v2_20261009/segment_01` | Generated from that corrected world and the accepted final image. GPT rejected a persistent near-prone tiger pose and new rod-like objects; the tool cannot represent these changes. | Still revision 1, one accepted history record, approximately 3 seconds |

The real acceptance/reconciliation/continuation path was exercised. It did NOT
produce two consecutively accepted clips. The final episode status is
`REGENERATE` for segment 01. That rejected segment's provisional corrections
and events remain observations and were not committed. Both native H3 renders
ran all 40 sampling steps; sampling took approximately 313 and 312 seconds.

The first guard trial was stopped before video completion to add comparison
against the established visual boundary. Its pending GPT plan was reused in
the fresh v2 trial. The v2 segment initially hit the floating-point event-time
bug documented in `video_accept_reconcile.md`; after fixing it, the same saved
GPT acceptance was committed with `--resume-review`. No alternate review was
sampled to obtain acceptance.

In accepted segment 00, the tiger endpoint correction relative to the planned
endpoint was `[0.320133, 0, -0.271112]` in Godot Y-up coordinates, approximately
0.42m. This cancels a simulated movement that did not appear in the video.
The player's root remained at the existing position. These are GPT estimates
from sampled images, not measured metric 3D ground truth.

## Code and data checks

Reviewed and extended the four side-session files: `whitebox/video_reconcile.py`,
`scripts/reconcile_whitebox_video.py`, `scripts/test_video_reconcile.py`, and
`docs/video_accept_reconcile.md`. Added the independent live driver
`scripts/run_video_reconcile_trial.py`, visualization/report builder
`scripts/summarize_video_reconcile.py`, and read-only snapshot audit
`scripts/audit_preserved_whitebox.py`. Extended `renderer/comfy_h3.py`,
`scripts/run_comfy_control.py`, and `scripts/test_comfy_renderer.py` for native
terminal RGB guidance. Updated README and this validation report. Existing
world schema, original experiment runners and installed inference code remain
unchanged.

Server regression tests: 108 unittest cases and 8 Godot contract cases passed;
Python compilation passed. Partial action fixtures verify advancing only the
attained state and continuing from it. They do not establish a live partial
action acceptance result.

`scripts/audit_preserved_whitebox.py` opens the original database read-only and
compares its entities/header with the pre-existing world snapshot. The original
`whitebox_persistent_20261009` world still has revision 2, approximately 5.625
seconds, 15 entities and two history records. Its audit is saved in
`video_reconcile_deploy_20261009/original_world_audit.json`.

The live chain audit checks history prefix preservation, equality of the next
plan's starting scene with the corrected endpoint, continuity of world time,
and use of the accepted final image as the next visual reference.
It also checks that the current DB remains exactly at the last accepted
endpoint, revision and time after the second clip is rejected.

## Inspection artifacts

The guard trial stores `segment_XX/plan`, `conditions`, `render`, `review`,
`corrected_preview`, `world.sqlite`, `chain_verification.json` and
`run_summary.json`. Review artifacts preserve the original plan, raw GPT
response, corrected trajectory, actual endpoint and before/after history.
`scripts/summarize_video_reconcile.py` creates the synchronized generated-video
and corrected-whitebox comparisons. `segment_00_accepted_comparison.mp4` shows
the accepted video and corrected coarse history. `rollout_sequence_labeled.mp4`
shows both generated clips with explicit ACCEPTED/REJECTED labels;
`boundary_continuity.jpg` compares the accepted endpoint and next clip's
boundaries. There is deliberately no two-clip `accepted_sequence.mp4` for this
run, because only one clip was accepted.

To generate the same pending intention in another fresh experiment:

```bash
cd /home/star/agentic_world
.venv/bin/python scripts/run_video_reconcile_trial.py \
  --output outputs/video_reconcile_guard_NEW \
  --scene outputs/motion_calibrated_20261009/scene.json \
  --reference outputs/h3_scene_20261008/frame_011.png \
  --initial-plan outputs/video_reconcile_guard_v2_20261009/segment_00/plan \
  --control depth
```

To regenerate inspectable reports from this run:

```bash
.venv/bin/python scripts/summarize_video_reconcile.py outputs/video_reconcile_guard_v2_20261009
```

## Remaining limits

- Guard-stance acceptance does not prove that controlled translation or complex
  combat works. The actual sidestep trial above failed.
- Actor root/yaw corrections are interpolated estimates. Procedural joints and
  held weapons are rigidly transformed, not reconstructed from the video.
- Ownership changes, drops, new entities, consequential pose changes, terrain
  edits and camera corrections remain unsupported by this reconciliation tool.
- The native terminal RGB API works, but our mask-shifted target image was not
  a suitable condition. The accepted guard trial uses first RGB plus full depth,
  not a terminal RGB guide and not exact endpoint pixel locking.
- Approximately 17 video frames are reviewed. Hidden events and all-frame
  geometric consistency are not established by this experiment.
- The continuation planner/reviewer also added a strict two-hand grip
  requirement, beyond the high-level request to retain the staff. That is a
  remaining overconstraint risk. The independent near-prone pose and new
  objects already justify this rejection; a grip adjustment alone should not
  generally be treated as a change of weapon ownership.

## Next priorities

1. Add explicit coarse pose states and support contacts, so standing/crouching
   transitions can be reconciled without pretending root offsets encode them.
2. Fit actor root trajectories to image landmarks/projection with uncertainty,
   and validate better terminal conditions instead of mask-shifted RGB targets.
3. Separate requested action outcomes from optional hand/cloth/body acting in
   the planner and reviewer; then test a real partially accepted locomotion
   action and its continuation.
