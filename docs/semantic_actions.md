# Coarse executable actions

The agent chooses action phases, durations, destinations and endpoint velocities.
The tool computes trajectories and validates geometry. H3 supplies limb animation;
the whitebox does not encode a gait skeleton.

- `move` with `gait=walk/run/crouch` and `end_velocity` uses the existing Hermite
  root trajectory. Actual sampled trajectory speeds are checked against the gait
  limit, not just the endpoint. Crouched movement needs a completed crouch phase.
- `posture` smoothly changes the coarse body height over its whole duration.
  Height factors are shared by Python collision and Godot RGB/depth rendering
  through `assets/gta/catalog.json`: standing 1, crouched .65, low_guard .4.
  This is a compressed occupancy silhouette, not an anatomically posed crouch.
  Rising into an overhead obstacle fails the numerical plan.
- `jump` uses a ballistic vertical arc, a bounded supported landing point and
  explicit horizontal endpoint velocity. Flight duration is .5..1.2 seconds;
  it determines launch velocity and height under gravity 9.81m/s2. Landings may
  connect directly to the next move. Planned takeoff/landing events are not facts
  about a generated video.

Airborne collision uses a rounded foot volume plus the upper body, with 5mm
contact tolerance. A flat bottom box incorrectly intersected sloping GTA ground
at 0.002s after takeoff. The rounded-foot correction was tested on sloped ground,
actual recorded collision meshes, walls and ceilings; increasing jump height is
not used to bypass that numerical defect.

The current jump must land within its segment. Airborne accepted endpoints,
prone crawling, arbitrary climbing and detailed contact forces are not supported.
NPC patrol/guard policies remain their existing coarse movement policies; adding
these player action primitives does not create a complete NPC animation system.
Velocity continuity is supported; acceleration/jerk continuity and natural H3
motion are not established by these tests.

## Actual agent test

`scripts/prepare_semantic_trials.py` creates two isolated scenes using the
recorded GTA street surfaces. The beam and barrier are explicitly AGENT_CREATED.
The real Pi agent was asked to author three-second plans via the registered
world tool. No historical database or accepted video was altered.

The crouch plan passes under the beam and stands beyond it. The jump plan goes
2m over the barrier, lands with forward velocity, continues another metre and
ends at rest. The agent's first run exposed the sloping-ground collision bug;
it was paused for an implementation fix. A subsequent agent session passed the
jump plan. This is an engineering debugging result, not an uninterrupted
autonomous video-world rollout.

`scripts/compile_semantic_trial.py` runs the checked compiler and writes RGB
whitebox video, depth controls and a timestamped review sheet.
`scripts/audit_semantic_trials.py` replays both plans, checks compiled snapshots,
73 video frames, moving landing continuity and unmodified accepted world history.
Results live in `outputs/semantic_actions` locally and on the server.

The original jump test camera cropped the apex. `jump_framed` preserves that
original result and creates a separate scene with a wider review camera. Entity
trajectories are asserted identical to the agent-authored jump plan; the camera
adjustment was made by the test operator, not autonomously by the agent.
Use `jump_framed/compiled` for the full-body jump controls.

Seventy action/geometry/world/reconciliation tests and seven Pi tool tests pass.

No H3 generation was run in this test. Good geometry and visible control changes
do not establish natural crouching or jumping in generated RGB video.
