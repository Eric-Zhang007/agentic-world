# Accept video, reconcile its segment, then continue

Update 2026-10-10: the same path now also accepts evidence-backed coarse
`pose_corrections` for standing, crouched and low_guard, plus grip selection
without changing ownership. See `posture_reconcile.md` for the live partial
acceptance/continuation result. The October 9 root-only trial results below
remain historical evidence; they have not been relabeled as posture tests.

This independent entry point implements the revised acceptance order.
It does not change the older action runner or historical experiments. The
side-session version was local only; the main-session integration below adds
server deployment and a separate real-video trial, without rewinding any DB.

## Main-session review and integration

Review confirmed that the existing `objects`, `world_meta`, `edits` tables and
Godot Y-up adapter can be reused without a schema migration. `BEGIN IMMEDIATE`,
revision comparison, deterministic replay of the pending plan and history append
already protect against stale commits and rewriting accepted segments.

Issues repaired before the live trial:

- Visible actors omitted from corrections previously inherited their planned
  endpoint silently. Every actor now needs CORRECTED, CONFIRMED_PLAN or
  UNOBSERVED_SIMULATED assessment with evidence. CORRECTED requires its track.
- Duration-only video checks could accept wrong frame mapping. The CLI now
  checks decoded presentation timestamps and records the exact video-to-plan
  index map. New renders use initial state followed by simulation ticks.
- A next action alone was not a renderable continuation. `next_plan/` now saves
  the committed input scene, agent action, full timeline, predicted endpoint,
  diagnostics, revision and boundary prompt. Partial actions continue their
  remaining goal before accepting a new goal.
- Wrapped yaw offsets interpolate on the short arc. Unsupported edit fields
  and explicitly reported unsupported state changes cannot be silently ignored.
- Corrected timeline phases are labeled `planned_phase`, not observed events.
  Accepted video events and actor state assessments are stored separately.
- An event at exactly 3 seconds initially failed because accumulated simulation
  time was 2.999999999999998. Endpoint validation now allows only 1e-8 seconds
  of roundoff and clamps that roundoff to the segment boundary. Later events
  remain invalid. The original GPT review is retained unmodified.

The required Chinese sentence is included verbatim in `BOUNDARY_PROMPT`:
“起始和结束状态应与给定条件一致，中间按动作描述自然丰富；收势是在目标终态稳定下来，不意味着回到起始位置。”

## Renderer boundary interface

`ComfyH3Renderer.render(..., first_frame=..., anchor_first=True, last_frame=...)`
uses installed native `MiniMaxH3AddGuide` at 0 and -1. Resolved frame indices are
recorded in worker metadata. This is actual image-guide conditioning, not a
denoising `control_end` parameter or a claim of exact pixel locking. Omitting
`last_frame` preserves the previous renderer behavior. Installed Comfy/H3
source, environment and weights remain untouched; only project wrappers change.

The isolated live trial prepares clean Godot depth for the complete pending
trajectory. Its optional `edited_rgb_target` mode also prepares an explicitly
labeled edited target image for the terminal RGB guide. Actor masks are
translated using projected planned root displacement;
the thin weapon uses a hull to prevent fragmented masks. The image has known
inpaint/background artifacts and is reviewed BEFORE generation. It is a target
condition, NEVER an observed frame or accepted world fact. Generated videos
must independently pass the acceptance/reconciliation path.

New actions get extra settling ticks if necessary to satisfy H3's 17k+5 frame
rule with the initialized state included. No previous accepted timeline is
resampled, stretched, or overwritten. Clip file duration includes one sample
period beyond its last timestamp; world time follows the aligned state ticks.

## Actual behavior

1. Prepare a pending whitebox plan without committing its endpoint. Its prompt
   explicitly asks for matching boundary states and free intermediate acting.
2. Render that plan using the existing renderer/control pipeline. The review
   CLI consumes an existing video; the separate live-trial driver also invokes
   H3 generation, using the installed native conditioning interface.
3. Review the prior accepted final frame (or initial source) followed by the
   new chronological video samples. Check that the new initial frame preserves
   established state. The agent can accept a
   useful partial action, describe trajectory corrections and record events.
4. Apply corrections to the pending segment, append its accepted history and
   commit its corrected endpoint/time in one SQLite transaction.
5. Only then ask the agent to plan the remaining action, using the corrected
   world and the accepted video's final frame. Completed actions wait for a
   next goal, unless `--next-goal` was explicitly supplied.

The previous accepted history is never silently replaced. An accepted partial
segment DOES advance time and world position. Rejection changes neither.

An interrupted pre-commit CLI review can be resumed with `--resume-review`,
using the same world, pending plan, video and output arguments. It reuses the
saved GPT response, checks the pending timeline and video alignment, and refuses
an already finalized status. It is not a way to turn REGENERATE into ACCEPT.
After a successful commit, `run_video_reconcile_trial.py --continue-trial`
skips accepted segments and continues their stored next plan. It neither
reinitializes that database nor generates over an existing rejected segment.

## Commands

From the project root, using the existing configured environment:

```bash
python scripts/reconcile_whitebox_video.py \
  --world path/to/world.sqlite --prepare-goal 'Sidestep, sweep once, and settle at the new position.' \
  --output outputs/pending_segment

# Render scene.json/timeline.json through the existing Godot + H3 pipeline.
# Use render_prompt.txt. The video and pending plan must share a time axis.

python scripts/reconcile_whitebox_video.py \
  --world path/to/world.sqlite --plan outputs/pending_segment \
  --video path/to/generated.mp4 --output outputs/reviewed_segment
```

Do not use an already committed historical segment as the pending plan.
The script refuses that case rather than rewinding a live world. In particular,
the old ablation clips are not ready-made live continuation inputs.

## Representation limits

- Agent estimates correct existing actors' root positions and yaw at sparse
  keyframes; corrections are interpolated over the planned trajectory.
- Joints and attached weapons receive the same rigid correction. These poses
  are still coarse proxies, not recovered video skeletons.
- Unseen NPCs retain simulated behavior. Static terrain and entities persist.
- Observed events and residual state notes enter the accepted segment record
  and continuation planner; they do not automatically create health physics.
- New entities, dropped props, changed ownership, arbitrary pose changes,
  camera corrections and static edits are not represented by this first tool.
  The review prompt must identify these as unsupported, not falsely claim
  they were reconciled. Extending these edits is separate work.
- Geometry diagnostics are recorded, not used as a hidden confidence gate.
  Agent acceptance is the decision; stale revisions, malformed coordinates
  and rewritten previous history remain ordinary data-integrity errors.
- Generated-video interpretations are explicitly labeled. They are accepted
  episode history, not independent evidence of an external physical world.
- The CLI samples approximately 17 frames, not every frame. Video judgment
  remains fallible. No numerical confidence threshold is required for acceptance.

## Verification performed

Local command:

```bash
python -m unittest scripts.test_video_reconcile scripts.test_whitebox_world scripts.test_whitebox_appearance
```

39 tests passed, including nine new reconciliation tests. The partial-progress
fixture plans a 0.6m step, accepts a corrected 0.3m step, advances world time,
and verifies the next simulation begins at 0.3m. Tests also cover trajectory
and held-weapon consistency, rejected clips, stale/replayed segments, preserved
history, invalid events, explicit endpoint requirements and plan tampering.
These are implementation tests, not evidence of a new real H3/GPT rollout.

The preceding 39-test statement describes the side conversation only. The
main-session live-trial result must be read from its separate status and
`chain_verification.json`, not inferred from these unit tests.

## Fresh server experiment

```bash
cd /home/star/agentic_world
.venv/bin/python scripts/run_video_reconcile_trial.py \
  --output outputs/video_reconcile_NEW \
  --scene outputs/motion_calibrated_20261009/scene.json \
  --reference outputs/h3_scene_20261008/frame_011.png
```

This initializes a new experimental DB from the disclosed inferred scene. It
does not clone an accepted history and then overwrite it. Original world and
ablation outputs are read-only inputs. Each segment saves its pending plan,
conditions and their review, actual native H3 generation, video review,
corrected trajectory, world snapshots and before/after history. The next
render reads `review/next_plan`, which is created only after commit. Corrected
Godot previews permit visual inspection of the accepted coarse history.

The default `--control depth` uses a native frame-zero RGB guide and the full
whitebox depth sequence, including its initial and terminal states. It does
not supply or claim to lock a terminal RGB image. The optional
`--control edited_rgb_target` additionally calls the installed native
`MiniMaxH3AddGuide` with index `-1`; resolved indices are recorded in generation
metadata. The first live trial using that option reproduced defects from the
composited target image and was rejected without advancing the world. Guide
API support is not evidence that a synthetic target image is a good condition.

History stores the original action as planned intention, with separate actual
completion, interpreted events, and the video review. Corrected timeline
`planned_phase` labels remain prediction metadata. They must not be interpreted
as proof that every planned phase actually occurred. The next review receives
this distinction explicitly; it does not merely see a list of planned actions.

Limits remain: 3D root/yaw corrections are agent estimates from sampled video,
not recovered metric ground truth. Joint and weapon geometry receive rigid
corrections; dropped weapons/ownership changes are unsupported. Offscreen NPC
motion retains simulated provenance. Neither successful generation nor an
accepted clip alone proves the two-segment closed loop; inspect the chain audit.
