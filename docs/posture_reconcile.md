# Persistent coarse posture and support

This extends the existing whitebox, without a database migration or a new scene
engine. The installed H3/Comfy environment and weights are reused. Old actors
without posture fields keep their original standing rig and motion behavior.

## State and execution

Actor fields live in the existing `attributes.whitebox` payload:

- `posture_weights`: nonnegative weights summing to one over `standing`,
  `crouched`, and `low_guard`. Mixtures represent coarse transitions.
- `support_contacts`: derived settled support points. Standing/crouched use
  the feet; completed low_guard also uses both hands/forepaws. These describe
  geometric template constraints, not measured contact forces.
- `grip`: `both`, `left`, or `right`, meaningful for an actor holding a staff.
  It selects procedural grip anchors; it does not change the staff's `holder`.
  The legacy default of both hands is a rig convention, not proof of visual
  hand count. Natural grip adjustment alone is not a release or ownership change.

The existing rig blends body/limb anchors and uses its two-bone IK. Foot and
hand contact centers account for the rendered joint radius. Root position and
yaw remain separate. Godot previews and ControlNet depth render the resulting
joint geometry, rather than merely displaying a posture label.

A new action phase executes an explicit pose transition:

```json
{"type":"posture","target_posture":"low_guard","duration_sec":3,
 "criterion":"Both forepaws contact the courtyard while the actor remains alert."}
```

Wait preserves the current posture. A crouched/low actor must rise before
locomotion; crouched walking and crawling are not implemented. NPC translation
is held while these support postures persist, rather than sliding the actor
under planted contact points. The existing guard/patrol policy is retained and
world time still advances. Four-contact low_guard cannot simultaneously hold
the staff in this template vocabulary.

## Video reconciliation

The real reviewer can now supply `pose_corrections` separately from root/yaw
corrections. Each names an existing actor, supplies evidence, and gives
chronological planned-frame indices with a posture and optional grip. An
explicit endpoint is required. Weights interpolate from the established
initial state through these keys. Grip changes take effect at explicit keys;
finger-level transitions are renderer detail.

CORRECTED actor assessments must correspond to a root track, a pose track, or
both. Reconciliation rebuilds the affected actor's coarse joints and held-staff
anchors, appends the accepted event/history record, and commits its attained
state. The next plan reads that committed state and the accepted video's last
image. An accepted crouch does not silently become the requested low_guard.
Partial acceptance continues the remaining goal before any unrelated next goal.

The reviewer and planner distinguish weapon ownership from hand count. They
must not invent a two-handed constraint when the requested outcome is simply
to retain the weapon. New props, dropped weapons, transfers, unconsciousness,
damage and poses outside the vocabulary remain unsupported by this tool.

## Server trial on 2026-10-10

`outputs/posture_live_20261010` is an explicit SQLite backup branch of
`outputs/video_reconcile_guard_v2_20261009/world.sqlite`. It inherits the one
accepted historical segment at revision 1 and approximately 3 seconds. The
parent is opened read-only and its history is preserved as a prefix.

The first new real H3 segment requested a three-second lowering to low_guard.
GPT accepted the video but marked `action_complete=false`: the forepaws remained
above the paving. Its pose corrections replaced the intended endpoint with
`crouched`, two-foot support. The branch advanced to revision 2 and approximately
6 seconds. It recorded defensive lowering, not damage or defeat.

The agent then planned from that actual crouch: two seconds to finish lowering,
followed by 3.125 seconds of holding after alignment to native H3's frame grid.
The continuation therefore has 124 video frames, representing an initial state
and 123 state ticks. It did not substitute the originally intended low_guard
endpoint or start an unrelated user action.

The second real H3 clip was also accepted, with `action_complete=true`. GPT
identified both forepaws on the courtyard at approximately 1.29 seconds, and
corrected the coarse transition timing while preserving the earlier crouch
history. The resulting low_guard has both feet and both hands as support points.
The branch ends at revision 3, approximately 11.125 seconds, with three history
records: one inherited and two newly accepted. The parent remains at revision 1.

Thus this run exercised actual partial acceptance, correction away from the
planned endpoint, continuation from the corrected pose, and subsequent action
completion. The next waiting plan retains low_guard in simulation; a third H3
hold clip was not generated. The two new clips have 73 and 124 frames, roughly
8.2 seconds of playback including boundary samples.

Timing is read from the trajectories in the audit. The second action's nominal
lowering phase lasts 2 seconds, while its simulator reaches low_guard at 1.5
seconds and settles; the accepted track reaches it at approximately 1.29
seconds. The review's comparison to "two seconds" refers to the nominal phase,
not the actual first full-contact tick. The raw review is preserved unchanged.

For final per-segment outcomes, consult `posture_state_report.json`, raw
`segment_XX/review/review.json`, and `chain_verification.json`. Test success or
a condition-preview acceptance is not equivalent to actual video acceptance.

## Commands

Server checks passed: 116 unittest cases plus 8 Godot contract cases (124 total).
The posture-specific tests cover persistent support, lowering/rising, hand/foot
geometry through the transition, invalid state, unsupported crouched locomotion,
single-handed staff retention, and a pose-correction commit followed by another
simulation with preserved history. Both archived October 9 guard plans replay
exactly under the updated simulator. The parent DB still matches its original
snapshot at revision 1 and approximately 3 seconds.

```bash
cd /home/star/agentic_world
.venv/bin/python scripts/run_posture_trial.py \
  --parent outputs/video_reconcile_guard_v2_20261009/world.sqlite \
  --output outputs/posture_NEW

.venv/bin/python scripts/summarize_posture_trial.py outputs/posture_live_20261010
.venv/bin/python -m unittest scripts.test_posture scripts.test_video_reconcile scripts.test_whitebox_world
```

The live driver first renders actual whitebox previews and asks GPT to review
them. It then generates H3, reviews sampled actual video, and only commits after
acceptance/reconciliation. Default two-segment trial budget is not a declaration
that the requested action finished; inspect `action_complete` and the pending
continuation. All output directories must be new.

## Limits

These are agent-selected pose templates, not fitted or metrically recovered
skeletons. Exact limb angles, contact forces, finger grips, arbitrary prone or
injured poses, articulated obstacle collisions and moving crouched gait are not
implemented. Existing rest dimensions remain conservative collision bounds.
The initial-image scene builder does not automatically infer the new posture
fields; this trial exercises planned and video-reconciled posture maintenance.
Unseen NPC behavior remains simulated, with that distinction recorded.
The rendered continuation also becomes darker/softer; appearance consistency
was not solved or separately benchmarked in this posture experiment.
