# Continuous Motion Rollouts

The ordinary move implementation eases into its endpoint during the first 75%
of a phase. Existing archived plans retain that behavior for deterministic
replay. New move phases can specify `end_velocity: [vx, 0, vz]` in world m/s.
Their cubic Hermite root path inherits the accepted actor's `motion.velocity`
and uses the entire duration. A moving actor cannot jump to a wait: an explicit
move ending at zero velocity is required first. Camera interpolation and NPC
policies remain unchanged. This is coarse root motion, not a biomechanics model.

For explicit follow shots, a phase may set `camera_follow_actor` to an existing
actor ID. Both camera position and target inherit that actor's actual simulated
displacement each tick, preserving the initial relative framing. This excludes
`camera_delta`; existing camera-delta interpolation is unchanged. Following
does not guarantee unseen background accuracy, and video review must distinguish
camera-induced image flow from retained travel against the ground.
`camera_distance_delta` adds a smooth outward dolly of 0..5 metres along the
initial viewing axis over the full phase. It preserves the initial camera and
can combine with actor following to provide additional framing clearance.

The evidence runner now reviews nine chronological proxy samples rather than
only initial/middle/final views. Weapon `proxy_contacts` are explicitly described
as weapon-to-actor events, not walking contacts. A required full-body view must
not be approved when feet or body are cropped. `--prepared-plan PATH` reuses an
uncommitted plan only after exact current-state and deterministic replay checks;
the original artifacts remain untouched.

`skills/render-rollout/SKILL.md` is loaded by `agent/render_contract.py` when
`scripts/run_evidence_rollouts.py` compiles each video prompt. The same plan
produces depth conditions and the prompt contract. Artifacts include the exact
skill text, contract, generated prompt and permitted elaborations. This is a
working explicitly loaded skill, not a general dynamic skill-discovery harness.

The review receives dense opening/closing samples and the prior video's tail.
Motion-enabled actors require `motion_assessments` with actual estimated
endpoint velocity, locomotion/stationary phase, evidence and uncertainty. A
stopped generated actor is committed as stopped even when continuous movement
was planned. Intermediate root velocities are derived from the accepted coarse
trajectory; they are not optical or metric measurements. Endpoint estimates can
disagree with that trajectory derivative and remain explicitly uncertain.

## Native Historical Prefix

The installed Comfy `MiniMaxH3AddGuide` accepts a short clip with 17*k+5 frames.
The wrapper exposes `prefix_video`, validates exact 24 fps and frame count,
and uses it instead of a single frame-zero guide. It does not silently shorten
or resample the prefix. Depth conditions prepend the matching archived corrected
world states, then future planned states. Previous history is never rewritten.

For 124 generated frames and a 22-frame historical prefix:

- Raw output indices 0..21 describe accepted past up to the current boundary.
- Index 21 is retained as the new clip's initial state.
- Only 102 new ticks, or 4.25 seconds at 24 fps, advance the world.
- The reviewed continuation has 103 frames. The full raw generation is kept.

Native guide conditioning does not guarantee identical pixels or continuous
velocity. The review compares the new boundary against the established frame
and historical motion samples. It may reject despite numerically correct time
alignment. Reference images and control frames are currently resized to the
existing 1024x768 canvas.

## Server Command

Run in `/home/star/agentic_world` with a new output directory and an available
GPU. The first goal must explicitly require movement through its final boundary;
the second goal continues that movement. Use concrete goals suitable for the
existing scene, with enough space for the entire route.

```bash
sh scripts/with_proxy.sh .venv/bin/python scripts/run_evidence_rollouts.py \
  --parent outputs/refined_rollouts_20261012/world.sqlite \
  --output outputs/continuous_motion_20261012 \
  --prefix-frames 22 --segments 2 \
  --goal "Begin a slow sidestep, retaining the staff; still move at the boundary." \
  --goal "Continue the same sidestep without pausing at the video boundary."
.venv/bin/python scripts/verify_continuous_rollout.py outputs/continuous_motion_20261012
```

Read `motion_verification.json` separately from `chain_verification.json`: valid
history bookkeeping does not prove good motion. Unit tests and native interface
support also do not establish a successful real two-video boundary.

## Interpreting a Partial Acceptance

The first live segment on October 12 retained less displacement than planned
and stopped at its end. The reviewer accepted useful progress, changed the
root trajectory and endpoint, explicitly stored zero velocity, and kept the
original target in its continuation goal. That is evidence for actual-state
maintenance, and a failure of the requested uninterrupted movement. These
claims must remain separate even if the next clip is visually attractive.

Very short displacement over a long window can resemble stance adjustment.
The next motion-control comparison should use a clearly visible, natural travel
distance and an unobstructed route; otherwise endpoint speed is hard to assess.
Do not resolve this ambiguity with arbitrary optical-flow thresholds or by
requiring the generated skeleton to copy the procedural rig.

Old accepted records without a motion field are treated as legacy stationary
boundaries by the new planner. Use that convention only when previous video
evidence supports rest, as in this test's parent wait segment. Existing moving
legacy histories require an explicit evidence-backed motion interpretation
before testing velocity continuity; absent data is not proof of rest.
