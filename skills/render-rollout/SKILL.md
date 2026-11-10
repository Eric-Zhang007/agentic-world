---
name: render-rollout
description: Compile a persistent whitebox action into an H3 video prompt, preserving accepted history, key events and moving boundaries while allowing natural intermediate animation. Use before every video rollout and when repairing generation mismatches.
---

# Render the Same World

## Paired History Inputs

The current Pi renderer uses epoch-15 LoRA and no depth ControlNet by default.
Pass `db` in the render tool arguments so it can retrieve historical paired
views near the planned endpoint. Use `memory_query` to inspect candidates.
Author action and camera prose without Picture/Video/Object tags. The renderer
owns the bindings: with history, Picture 1 is historical RGB-left/whitebox-right,
Picture 2 is the current first RGB, Video 1 is the planned whitebox video.
Without history, Picture 1 is the first RGB. Do not add legacy object crops.

For a new initialized world, use `memory_capture` with its original RGB once
the initial scene preview is acceptable. Accepted video commits capture a pair
from the corrected terminal scene, never from the predicted endpoint. Check
the commit's paired_memory receipt; CAPTURE_FAILED means the world committed
but memory needs a retry, not that the rollout was rejected. Historical frames
describe prior appearance, not current ownership, pose or location.

`geometry_propose` runs RGB reconstruction and segmentation into a candidate
scene. Inspect its camera preview, ground support, occlusion and inferred scale
before initialize. It is not a world update or proof that hidden geometry is
correct. Existing-world expansion still needs region alignment and review.

Read the current accepted state, user intention, action plan, boundary contract,
and available appearance references. Conditions and prompt must describe the
same world. A plan is a prediction, never evidence of an observed outcome.

Produce a concise render prompt and explicit lists of required outcomes,
allowed elaborations, and uncertainties. Describe visible identities rather
than internal IDs or whitebox colors. Follow the planned camera. Keep numeric
world coordinates, camera extrinsics and procedural joint angles out of the
video prompt: translate them into visible relations and movement. The geometry
condition already carries them. Distances and qualitative speed may clarify
the requested movement.

Do not promote incidental representation fields into extra action requirements.
A stored two-handed grip or a procedural 55-degree weapon angle does not require
that exact grip or angle throughout the video. Require weapon retention and
safe clearance; allow one/two-handed adjustment unless the user explicitly
requests a particular grip or an interaction depends on it. Do not describe
an offscreen NPC in the render prompt merely because it is in world memory.

- Preserve established layout, identities, held objects, support state, and
  causal history. New scenery and consequential events require a world plan.
- Match coarse routes, start/end state, contact targets and event order.
  Do not demand frame-by-frame skeleton imitation. Staff angle, grip adjustment,
  fabric, facial acting, anticipation and balanced footwork can be enriched
  unless they affect contact, support, ownership, or obstacle clearance.
- A moving endpoint is not a stop. Continue direction and speed through the
  video boundary; no final pose, pause, return, or automatic recovery. If the
  action ends at rest, settle at its target rather than returning to its start.
- A generation boundary is only a time slice. Do not append wait/recover,
  decelerate an actor, or ease the camera to rest merely because a clip ends.
  For continuing locomotion, specify move.end_velocity. For continuing camera
  motion, specify camera_end_motion with translation_velocity, orbit_velocity_deg
  and distance_velocity, in world units/second and degrees/second respectively.
  Match the next plan to the accepted boundary motion. Only stop when the task
  requires it. A follow camera inherits the actor's translation.
- GTA rollout conditions use a rigid multipart actor, heading and routes,
  without limb animation. Posture changes scale its coarse height, consistently
  with body clearance. They do not prescribe knee angles or a precise skeleton.
  Translate gait and jump phases into natural walking/running/crouching or a
  takeoff-flight-landing sequence. Preserve obstacle clearance and landing
  location; let H3 supply limb coordination. Planned jump events are not evidence
  of a successful generated jump. Inspect actual takeoff, clearance and landing.
  Let H3 invent gait and secondary motion. This does not remove held objects,
  contact targets, clearance requirements or causal events.
- Timestamped image samples are not video playback. A static leg silhouette
  moving across the ground is not proof of walking. Inspect dense temporal
  crops when gait matters; report coarse progress separately from natural motion.
- A short historical video prefix provides motion evidence. It is already
  accepted history, not a new action, and must not advance the world clock.
- Do not claim depth or one reference frame guarantees speed continuity.
  Inspect chronological boundary windows, consulting motion tools if helpful.
- After viewing the rollout, explicitly assess actual endpoint motion and
  remaining action. Accepting useful partial progress must commit that actual
  state before the next plan. If stopped, record stopped; never invent velocity
  to make a continuity test pass. Preserve previous accepted segments.

Required boundary wording: start and end must match the supplied conditions;
enrich intermediate motion naturally. Recovery means settling at the target,
not returning to the start. Apply recovery only when the endpoint is at rest.
