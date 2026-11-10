# Optional evidence and coarse-world acceptance

## Scope

The whitebox retains consequential state and causal history, not a reconstruction
of every rendered surface or a contact-force simulation. Planned intermediate
motion can differ from the accepted video. Acceptance must reconcile the actual
coarse trajectory, events and endpoint before the next plan; old accepted history
remains immutable.

Perception tools produce evidence, not automatic world edits. The new motion
pilot is an independent read-only experiment, not yet an agent skill runtime and
not a replacement for the transactional video acceptance path.

## Existing capabilities and limits

- `scripts/reconcile_whitebox_video.py` sends approximately 17 timestamped images
  to GPT. It does not send continuous video or tracked motion.
- `perception/detector.py` reuses GroundingDINO, SAM2 and DINOv2. Tracks only link
  adjacent sampled frames; an empty frame loses continuity. Detection failure
  must not become an object disappearance fact.
- `matching/matcher.py` provides assignment and ambiguity handling. Match costs
  are not calibrated probabilities or independent evidence of world geometry.
- `whitebox/motion.py` checks proxy movement, coarse obstacle clearance and
  approximate weapon proximity. Proxy contact is not a visually established hit
  or damage event.
- The current action/reconciliation vocabulary cannot commit a dropped staff or
  ownership transfer. Those tests remain unsupported until the state transition
  is implemented. Existing acceptance must not silently discard them.

The cached `motion_depth_ablation_20261009/evaluation/dense_s43` failure is useful:
matched actors fall from two to one, tracking identity changes, and the visual
review rejects the retained endpoint and attack. A box metric alone cannot
resolve this case.

## Motion evidence pilot

`perception/motion_evidence.py` examines every adjacent decoded frame in a requested
time interval. It uses forward/backward feature tracking and a robust dominant
image transform. It emits tracking failures, timestamped motion measurements,
high-residual intervals and chronological images. It streams decoded images,
does not load models, and does not modify world databases.

The dominant transform is NOT assumed to be the camera. Foreground movement,
articulation, occlusion and lighting can contaminate it. All distances are image
pixels at the reported analysis width, not metres. No identity or contact claims
are emitted.

Run on the server:

```bash
cd /home/star/agentic_world
.venv/bin/python scripts/test_motion_evidence.py
.venv/bin/python scripts/evaluate_motion_evidence.py \
  --video outputs/posture_live_20261010/segment_01/render/controlled.mp4 \
  --output outputs/my_motion_inspection \
  --config configs/default.yaml
```

Omit `--config` for local motion analysis without GPT. Use `--start` and `--end`
to inspect a specific event. Output directories must be new, protecting prior
results. The GPT comparison is currently specific to the tiger low-guard test.

The pilot compares 17 evenly spaced images against boundary images plus a dense
local window and motion summary. This is exploratory: it changes both sampling
and numerical evidence, so it cannot isolate optical-flow benefit. It has no
independent metric ground truth. The next controlled evaluation needs equal
image budgets, repeated judgments, and separate dense-only / dense-plus-motion
arms before claiming accuracy improvement.

## Skill contracts after evaluation

### Pilot results, 2026-10-10

Real server runs are in `outputs/motion_evidence_20261010/{crouched,low_guard}`.
Both reuse actual H3 videos from `posture_live_20261010`; no video was regenerated
and no world or previous review was edited. Two synthetic tracking regression
tests passed. Existing server OpenCV 5.0.0 and PyAV 18.1.0 were reused, with no
dependency or weight installation. Four real GPT calls completed.

| Actual clip | 17-image judgment | Dense window + motion judgment |
| --- | --- | --- |
| Partial crouch | Incomplete; forepaws above ground | Same conclusion |
| Low guard continuation | Forepaws grounded; hidden rear foot uncertain | Same conclusion |

The strict four-support completion question returned `null` for low guard in both
arms. This does not overturn the earlier accepted coarse posture or its immutable
history. It reveals a mismatch between an unnecessarily strict visual question
and the intended coarse state. Both reviews describe the visible low alert pose.
Neither is a new world acceptance decision.

For the continuation, onset estimates differ: the sparse review reports
0.625-1.0 s; the enriched review reports 0.625-1.125 s, with text suggesting
apparent contact around 0.875 s. These are uncertain image interpretations,
not ground-truth event times. The experiment does not establish a benefit from
optical flow or improve the earlier accepted timing estimate by fiat.

Future questions should separate coarse action completion, visible evidence and
unobserved support details. Escalate uncertainty only when it changes a causal
decision. More frames cannot reveal a permanently occluded surface.

### Proposed descriptions

Expose descriptions first; load details and execute only when needed. Reuse the
existing client tool-calling mechanism rather than rewrite the agent framework.

| Candidate skill | When useful | Output and limits |
| --- | --- | --- |
| Inspect motion interval | Sparse samples miss an onset, transition or apparent jump | Dense chronological evidence and image motion; no hit/damage verdict |
| Detect and match entities | Identity, unexpected object or disappearance is unclear | Candidates, masks, embeddings, ambiguity and tracking gaps |
| Check projection | Apparent arrival or position contradicts the plan | Image residuals with camera uncertainty; no invented metric depth |
| Audit accepted history | A new segment may drift from committed state | Boundary/state/relation differences and causal references |

All return SUPPORTED, CONTRADICTED, UNOBSERVED or UNSUPPORTED per claim, with
evidence timestamps and uncertainty. These labels are an evaluation contract;
they are not yet wired into the live agent. An occluded expected support point
need not force regeneration: preserve an explicitly inferred coarse state unless
the unresolved fact changes the action outcome. Do not demand new rendered camera
views as proof of what happened in an earlier video.

## Next acceptance scenes

1. Target arrival: use a visible landmark and coarse target region. Verify actual
   endpoint after a partial move and begin the next action there. Avoid unjustified
   centimetre-level precision from a single camera.
2. Obstacle avoidance: route around one persistent obstacle. Distinguish planned
   proxy clearance from visible passage; inspect an occluded interval only if the
   route or outcome is consequential.
3. Attack contact: inspect a short event interval and target response. Keep visual
   proximity, likely contact, and causally established damage as separate claims.
4. Drop: first represent held -> released -> resting, holder changes and event
   time. Then verify that later plans retain the prop at its accepted coarse
   location. Do not test a capability by merely rejecting every drop.
5. Persistent sequence: execute several user actions on one branch and audit
   actual endpoints, appearance references, ownership, NPC policy/state and
   unchanged history prefixes after every accepted segment. Offscreen NPC states
   remain explicitly simulated rather than falsely described as observed.

These cases are outstanding acceptance work. The earlier two-segment posture
trial is positive continuation evidence, not evidence that all interactions or
long-horizon persistence already work.
