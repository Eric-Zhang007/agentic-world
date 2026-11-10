# Coarse target arrival

This experiment continues the accepted courtyard world on an explicit SQLite
backup branch. The parent database and accepted history are not rewritten. It
tests arrival before attempting obstacle, contact or ownership transitions.

## Task

The player moves approximately 0.55 coarse-world metres toward screen-right and
retains the new position within a 0.18-metre target region. The target is an
annotation on existing paving, not a spawned object. Units come from the inferred
whitebox and are not an independent metric reconstruction of the video.

The tiger keeps its accepted low guard; the player retains the staff; offscreen
NPC policies continue. The next clip either continues an incomplete move toward
the SAME target, or waits at the actual accepted endpoint. It must not restart
from the planned endpoint or reset the target relative to a partial move.

## Implementation

- `whitebox/interaction.py`: target-distance diagnostics and a pre-commit check
  against internally contradictory arrival claims. Numerical proximity alone
  does not establish visual success.
- `scripts/run_arrival_trial.py`: branch, GPT plan, rendered whitebox inspection,
  native H3 depth control, video reconciliation, and actual-state continuation.
- `scripts/reconcile_whitebox_video.py --interaction-contract`: additional
  SUPPORTED / CONTRADICTED / UNOBSERVED arrival assessment. Partial progress is
  allowed. Only visual support plus consistent corrected state permits completion.
- `scripts/summarize_arrival_trial.py`: actual first/last frame panels with target
  annotation, corrected whitebox comparison, and history/endpoint diagnostics.
- `agent/interaction_evidence.py`: independent read-only tool-calling reviewer.
  It starts with sparse samples and may ask `inspect_motion_interval` for up to
  two seconds of dense frames and motion evidence, at most three calls. It uses
  the existing `AgentClient`, not a new framework. It does not commit world state.
- `scripts/inspect_arrival_entities.py`: optional cached GroundingDINO -> SAM2 ->
  DINOv2 boundary evidence using the existing pipeline and assignment functions.
  Saves masks, embeddings, observed/planned boxes and displacement differences;
  it does not establish continuous identity, foot-root motion or metric arrival.

The optional reviewer is an evaluation entry point. Its tool is described with
triggers, outputs and limitations, but this is not yet the full skill loader or
automatic detection/segmentation/matching skill suite. Its judgments do not
silently override the transactional acceptance review.

## Server Commands

```bash
cd /home/star/agentic_world
.venv/bin/python -m unittest scripts.test_interaction scripts.test_interaction_evidence \
  scripts.test_motion_evidence scripts.test_video_reconcile scripts.test_posture
.venv/bin/python scripts/run_arrival_trial.py \
  --parent outputs/posture_live_20261010/world.sqlite \
  --output outputs/NEW_ARRIVAL_RUN --segments 2 --duration 5.125 --control-strength 1.0
.venv/bin/python scripts/summarize_arrival_trial.py outputs/NEW_ARRIVAL_RUN
.venv/bin/python scripts/review_arrival_evidence.py \
  --video outputs/NEW_ARRIVAL_RUN/segment_00/render/controlled.mp4 \
  --contract outputs/NEW_ARRIVAL_RUN/interaction_contract.json \
  --reference outputs/NEW_ARRIVAL_RUN/segment_00/conditions/first.png \
  --annotation outputs/NEW_ARRIVAL_RUN/target_annotation.jpg \
  --output outputs/NEW_EVIDENCE_REVIEW
```

All new run/review directories must be absent. Inspect `arrival_report.json`,
`arrival_verification.json`, `chain_verification.json` and each raw `review.json`.
The optional reviewer preserves `agent/tool_trace.json` and any requested dense
inspection artifacts. No new dependencies, checkpoints or training are required.

## Interpretation

Proxy clearance and arrival checks only test the coarse plan/state. Rendered
motion and historical continuity still require evidence. A new optical-flow
summary is not an independent observation of the same video. A partial arrival
is not a complete task; a rejected rollout must leave the current world unchanged.
Two accepted clips do not establish long-horizon persistence or contact/drop support.

## Initial Real Result, 2026-10-10

`outputs/arrival_live_20261010` branched revision 3 at world time 11.125 seconds.
The whitebox plan and rendered preview were accepted for generation. The 73-frame
H3 clip used native first-frame anchoring, full depth, strength 0.6, 40 steps,
1024x768 and seed 61. Sampling took 313.8 seconds. All ten ControlNet injection
layers executed 40 times with nonzero measured residuals.

The generated character mainly shifted weight and adjusted the staff. Both the
transactional video review and independent sparse evidence review rejected
arrival. No new history was committed. The parent remained revision 3 with
three accepted historical segments and fifteen entities.

Real cached-model boundary inspection in
`outputs/arrival_evidence_20261010/entities_00` matched both actors at both ends:

| Actor | Planned box delta X | Observed box delta X | 2D delta error |
| --- | ---: | ---: | ---: |
| Player | 164.97 px | 10.91 px | 154.14 px |
| Tiger | 0.00 px | 0.78 px | 3.53 px |

Assigned track IDs remained the same. These approximate box measurements support
the failed-arrival interpretation, but are not root trajectories or metric 3D.
The detector loads existing cached models on GPU 0; H3 uses GPU 3. Processes release
their models after execution. No weights or dependencies were downloaded.

The independent reviewer also correctly rejected a stationary prior clip used as
a negative arrival example. In both completed evidence reviews it declined
temporal inspection because the endpoint separation was already clear. Tool
dispatch and returned image evidence are regression-tested, but these live calls
do not establish that dense temporal inspection improves accuracy.

`outputs/arrival_retry_20261010` reused the rejected plan, target, parent world and
seed, applied the agent's retry prompt, and raised control strength to 1.0. It was
also rejected for lack of retained travel, leaving revision 3 unchanged. Because
both the prompt and strength changed, it is a targeted retry rather than a
single-factor ablation.

The installed ComfyUI source
`/home/star/agentic-world-comfy/ComfyUI/comfy_extras/nodes_minimax_h3.py:266`
labels the trained length range as approximately 124-362 frames. The first two
trials used 73 frames; this UI/source hint motivated a separate 124-frame test,
not a conclusion that duration caused the failure. The same source's guide node
sets `guide_frames=1` for a one-image guide. The worker passes only one reference
image to the anchor; its repeated static `source_resampled.mp4` is a diagnostic
artifact and is not supplied as a reference-video condition.

The 124-frame diagnostic pair is `arrival_124f_20261010` (depth strength 1.0) and
`arrival_no_control_20261010` (strength 0). The latter reuses exactly the former's
prompt, reference, depth input, seed, resolution, frame count and step count; it
runs on a different idle GPU. It is a renderer comparison, not an accepted world
branch. Inspect the two generation metadata files to verify these settings.

`outputs/arrival_evidence_20261010/tool_call_smoke` additionally records a real
explicitly requested API tool call. The model selected 0.5-1.5 seconds, the tool
measured 24 frame pairs and returned nine chronological images, and the model
described idle/stance motion with image-space limitations. This validates actual
tool execution and returned visual evidence, not autonomous tool usefulness or
an accuracy improvement. The optional autonomous arrival reviews made zero
inspection calls because their endpoint evidence was already clear.

## 124-Frame Arrival and Control Comparison

The controlled 124-frame arrival was accepted, advancing its branch to revision 4
at world time 16.25 seconds. GPT explicitly confirmed the planned coarse endpoint
from the relocated feet and torso; it did not apply a nonzero root correction in
this case. Observed stepping/settling events were recorded, with the limitations
of sampled single-view interpretation. The timeline remains a coarse accepted
representation, not a measured per-frame reconstruction.

The no-control sample also moved right, but the independent reviewer classified
retained displacement as SUPPORTED and target arrival as CONTRADICTED: its final
placement was left of and farther up the image than the target. No world state
was committed from this diagnostic sample. Matching settings are recorded in
`arrival_no_control_20261010/comparison_metadata.json`; the controlled run had
40 calls at each of ten injection layers, while the no-control run had none.
Sampling times were 646.5 and 551.6 seconds, respectively, on separate GPUs.

The no-control worker correctly wrote `baseline.mp4`, but the old renderer wrapper
looked only for `controlled.mp4`. This artifact-name bug was fixed and covered by
two additional tests. The already generated video was recovered without rerunning
H3; original error metadata was retained.

The real perception comparison is also a metric counterexample:

| 124-frame condition | Observed box delta X | 2D box-delta error | Same assigned player track | Visual target assessment |
| --- | ---: | ---: | --- | --- |
| Depth 1.0 | 198.78 px | 58.28 px | Yes | Supported |
| No control | 134.95 px | 31.44 px | No | Contradicted |

A lower box-center error did NOT identify the successful target arrival. Pose,
box extent and tracking ambiguity contaminate this metric. Evaluate support/foot
placement against the floor and target context; do not promote box residuals to
authoritative root positions. This is one scene and one seed, not a general
ControlNet accuracy claim. Changing clip length also changes latent sampling and
planned timing, so the earlier failures do not isolate a duration-only effect.

## Hold Continuation and Preview Recovery

The pending hold starts from revision 4's accepted endpoint and lasts 5.125 seconds.
An initial preview review incorrectly applied arrival-motion criteria to this
wait phase and demanded visible proof of an offscreen NPC. That original rejection
is preserved in `segment_01/condition_review.json`.

`review_conditions` now distinguishes ARRIVAL from HOLD_AT_ACCEPTED_ENDPOINT,
provides alternate proxy views for occluded weapon geometry, and labels NPC state
as simulated. Re-review accepted the same pending hold without changing its scene,
action, target or accepted history. The separate result is preserved in
`segment_01/condition_review_retry.json`. A regression test covers the phase-purpose
distinction. The video-review prompt likewise explicitly permits retaining an
already attained target during a wait, without demanding new translation.

The recovery command used existing compiled conditions and refused any segment
that had already entered rendering or review:

```bash
.venv/bin/python scripts/run_arrival_trial.py \
  --parent outputs/posture_live_20261010/world.sqlite \
  --output outputs/arrival_124f_20261010 --control-strength 1.0 --resume-segment 1
```

This command is only for the recorded pre-generation rejection, not for overwriting
an already completed segment. Final continuation status is recorded in
`resumption_verification.json` and `chain_verification.json`.

## Final Outcome

The hold video begins consistently with the accepted arrival endpoint, but later
changes the camera projection: the monument, stairs and tiger shift relative to
the player. GPT returned REGENERATE with target retention UNOBSERVED. It explicitly
refused to disguise camera movement as actor-coordinate corrections. The current
reconciliation tool cannot store a corrected camera trajectory, so this segment
was not committed.

`integrity_report.json` verifies that the branch remains revision 4 at world time
16.25 seconds, with four historical segments including the three inherited ones.
All fifteen entity IDs, eleven static entities, staff ownership and the committed
camera are preserved. Current state equals the accepted arrival endpoint and the
continuation's initial plan. The original parent remains revision 3 at 11.125
seconds. The rejected hold advances neither world state nor history.

The motion tool measured all 123 adjacent pairs of the rejected hold. At analysis
width 640, median pair displacement was 0.761 px, versus 0.014 px for the earlier
fixed-view low-guard clip. This supports inspecting image-wide motion; it does
not recover metric camera motion or prove a unique rigid scene explanation.

Five real H3 videos were generated this round. Target arrival passed once and
actual-state continuation passed at the next clip's opening, but persistent
world/camera alignment did NOT pass across the full continuation. No claim is
made that the long-horizon acceptance criterion is solved.

The local/server source changes passed 45 relevant unit/regression checks across
interaction, optional evidence, motion, video reconciliation, posture, renderer
artifacts and preview purpose. These checks are separate from real video outcomes.
No dependencies or model weights were installed. GPU processes were released.

Review `rollout_sequence_labeled.mp4`: the first segment is accepted, the second
explicitly rejected. `segment_00_accepted_comparison.mp4` compares the accepted
arrival to its coarse world; `boundary_continuity.jpg` shows accepted final,
next initial and rejected next final side by side. `arrival_verification.json`
is the earlier interrupted runner snapshot; the final outcome is in
`arrival_report.json`, `resumption_verification.json`, `chain_verification.json`
and `integrity_report.json`.

The next priority is optional camera-motion/alignment evidence plus an explicit
coarse camera reconciliation path that preserves prior history. Camera motion
need not be intrinsically invalid when allowed by the goal, but it must be
represented before retaining world-space claims. Obstacle, contact, drop and
longer multi-action acceptance remain outstanding. The full skill catalog/loader
is also not implemented; the current deliverable is a real optional temporal tool
and reusable perception inspection entry points.
