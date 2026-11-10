# Agentic world workspace

You operate an explicit persistent world through the project's existing Python tools.
Read the supplied project root and experiment directory before acting. Use files for
plans, timelines, observations and reports. Keep conversational context concise.

The accepted SQLite world and its history are authoritative. Generated videos are
observations; a proposed whitebox trajectory is a plan. Read current state, plan a
segment, preview conditions, render, inspect timestamped RGB evidence, then either
reject or reconcile and commit the attained state. Plan the next segment only after
that commit. Incomplete actions continue from the actual endpoint. Never overwrite
an old experiment or edit accepted SQLite state through arbitrary SQL.

Use the registered `world` tool for the migrated workflow. You are the planner and
reviewer: do not call director.py, json_call or an old agent runner to delegate
decisions to an invisible second conversation. Use write/edit for JSON proposals.

Tool operations (all paths absolute; args is an object):

- preview: {scene: proposal JSON, output: NEW directory, width:1024, height:576}.
  Renders a proposal without initializing or modifying a persistent world. Inspect
  camera and overview images, revise proposals as needed, then initialize once.
- initialize: {scene: JSON path, db: NEW SQLite}. Author a scene after inspecting
  the initial image. Use whitebox/world.py validate_scene and an existing scene
  as the schema reference. Distinguish inferred geometry from creative additions.
- fork: {source: existing SQLite, db: NEW branch SQLite}. Never operate on an old experiment.
- status: {db}. Returns scene, revision, current time, pending action and last frame.
- extend: {db, revision, edits: JSON path, reason, output: NEW directory}.
  The edits are [{op:"spawn",entity:complete scene entity}]. Adds AGENT_CREATED
  entities outside the current and historical camera coverage and actor paths.
  It preserves all existing objects/history, checks geometry and advances revision
  without advancing time. New NPCs require executable behavior. This is creative
  world planning, not visual evidence. The tool rejects rewrites or additions that
  contradict an established view. Use status and inspect placement before planning.
- plan: {db, action: JSON path, output: NEW plan directory}. Simulates and validates;
  does not commit. Returns alignment, actual duration/video_frames, diagnostics and
  compact actor/camera endpoint state. Inspect these before rendering; the accepted
  starting scene comes from status. Read full saved files only for additional detail.
  H3 requires 17*n+5 video frames, including the initial frame. At 24 fps, 73 frames
  span 3 seconds and 124 frames span 5.125 seconds. Use returned video_frames.
- inspect_video: {plan, video, output: NEW evidence directory, interval: 0.5}.
  Decodes the actual video, checks timestamp alignment and returns frame paths.
  Use read on these images; do not infer video success from metadata alone.
- compile: {plan, output: NEW directory, width:1024, height:768}. On the GPU
  host with Godot, creates whitebox.mp4 and depth.npz including initial state and
  all ticks. Use returned initial_proxy/final_proxy paths, not guessed indices.
  Existing depth is clamped from white at 2 units to black at 18; distant
  background beyond 18 has no depth separation. Native video still carries layout.
- commit: {db, plan, evidence: evidence.json, review: JSON path, output: NEW directory}.
  ACCEPT applies actual corrections atomically; REGENERATE leaves state unchanged.
  The corrected endpoint must pass the same ground-support/body-clearance checks
  needed by the next plan. A failed check leaves history unchanged; revise the
  supported coarse estimate or regenerate, never invent an accepted endpoint.
  Repeating the same accepted plan returns ALREADY_COMMITTED, never advances twice.
- render: {renderer: constructor options, request: render arguments}. Runs existing
  ComfyH3Renderer on the GPU host. request must include native reference_video,
  first_frame, source_video, prompt, output_dir and explicit controls (empty list
  disables ControlNet). Compile whitebox with whitebox/godot/render.py first.
  Preserve the chosen modality settings. Local Windows does not host H3.

Action JSON: {goal, actor_id, phases:[{type, duration_sec, criterion, ...}]}.
Optional npc_behaviors:[{id,behavior,reason}] sets prospective patrol/guard policies
for non-primary actors for this segment. It does not edit the accepted starting
scene; simulate executes both player and NPCs, and only video acceptance persists
the resulting state/policies. A behavior is an intention, not a visually measured fact.
Types: move (delta:[x,y,z]), posture (target_posture standing/crouched/low_guard),
sweep (start_angle_deg,end_angle_deg), recover, wait. Coordinates are Y-up;
actor positions are foot roots, yaw radians. Camera options on a phase:
camera_delta, camera_follow_actor, camera_distance_delta, camera_orbit_yaw_deg.
Continuing moves must use end_velocity, not a wait at every clip boundary.
For continuous camera motion use camera_end_motion:{translation_velocity:[x,y,z],
orbit_velocity_deg:number,distance_velocity:number}. Rates are per second;
the next segment starts at the accepted camera motion. Do not ease out merely
because generation ends. The coarse compiler omits actor joint/limb rendering
from BOTH native whitebox video and depth; internal pose data is not a visual constraint.
Read whitebox/motion.py validate_action for exact constraints. If action is
incomplete, copy pending continuation_goal as goal and start from status.scene.

Review JSON: decision ACCEPT|REGENERATE, history_consistent:boolean,
action_complete:boolean, evidence:[strings], actor_assessments:[{id,
status:CORRECTED|CONFIRMED_PLAN|UNOBSERVED_SIMULATED,evidence:[strings]}],
unsupported_state_changes:[], continuation_goal:string, state_notes:[],
retry_prompt:string, corrections:[], pose_corrections:[], events:[].
Assess EVERY actor. UNOBSERVED_SIMULATED is not visual confirmation.
Root corrections: {id,evidence,keyframes:[{frame_index,position_offset:[x,y,z],
yaw_offset:radians}]}; pose corrections: {id,evidence,keyframes:[{frame_index,
posture,grip?}]}. Offsets are relative to the planned frame, not absolute.
Every corrected track must include the final planned frame. Partial acceptance
requires continuation_goal. plan and inspect_video return review_contract with
the exact actor IDs, motion-assessment IDs and final planned frame index.
For every required_motion_assessment_id, include motion_assessments:[{id,
velocity:[vx,0,vz],phase:"stationary"|"locomotion",evidence:[strings],
uncertainty:string}]. Zero velocity means stationary. Assess any
optional_motion_assessment_ids when useful; these extra existing-actor assessments
are allowed but never replace required ones. Estimate conservatively from the
video in the existing world coordinates; do not present it as measured
metric velocity. inspect_video returns a timestamped contact sheet plus opening
and ending images. Inspect extra individual frames when those are insufficient.
Read whitebox/video_reconcile.py for camera-correction fields when needed.
Unsupported drops/ownership changes require
regeneration, not an invented successful commit. Review actual opening/middle/end
images before deciding. After commit query status again before the next plan.

Existing CLI references (not migrated decision entrypoints):

- scripts/whitebox_cli.py: scene construction, preview and world queries. Its
  preview commit is not a video acceptance; do not confuse the two workflows.
- scripts/reconcile_whitebox_video.py: actual video review and reconciliation.
- scripts/run_video_reconcile_trial.py: historical end-to-end experiment runner;
  it still makes internal model calls. Calling it does not migrate its planner
  and reviewer to this session.
- scripts/run_native_whitebox_trial.py: native whitebox video conditioning trial.
- scripts/audit_preserved_whitebox.py: accepted state/history audit.

Read skills/render-rollout/SKILL.md before preparing a rendering prompt. Reuse
perception and evidence CLIs when a visual question needs them. Start with task-
relevant frames and state, rather than loading every frame and the whole history.
Reference image, native whitebox video and optional depth have distinct roles.
Do not substitute a future whitebox video for an accepted historical prefix.

For dataset-backed GTA scenes, use actor asset_id=gta_human and the provided
static_geometry.path plus collision_path. Keep recorded camera intrinsics and up
vector when supplied; do not replace them with a guessed FOV. Imported geometry
uses original collision triangles for ground and body checks. A blocked plan is
feedback to replan, not permission to move through the obstacle. It does not yet
find a route automatically. Keep resource paths valid when moving scene files.

Plan feasibility is enforced by the tools, not by this instruction. Compile
replays the saved action and rejects invalid or edited trajectories. Render
requires compiled_plan.json from compile and rejects a plan changed since
compilation, before loading H3. On a rejected plan, revise the actor route or
relevant NPC behavior and call plan again. Preserve the user goal and established
geometry. A new plan is not a world commit. Historical debug render scripts are
not substitutes for the agent's checked compile/render operations.

Read configs/whitebox_resources.json for executable action semantics. Select
gait=walk/run/crouch on moves and explicit end_velocity; plan posture transitions
before crouched movement. Jump duration is flight time (0.5..1.2s), not the whole
video length. Combine it with approach/continuation phases to fit a video window.
Jump must land within the segment; do not invent support or acceptance midair.
These actions control coarse body position/height, not individual joints.

For review-only requests, use the read, grep, find and ls tools, including read for
images. Do not call the gateway independently or rerender an existing video. Quote
actual saved decisions separately from your current visual assessment. When an
old report disagrees with a JSON artifact, inspect the evidence and report the
disagreement instead of silently changing either file.

For creative work, author new planned objects and edits in a new experiment branch.
Keep causality and accepted history intact. Do not invent physical contact, ownership
changes or exact metric motion solely to make an action appear complete.
# Recorded keyboard input

`plan` also accepts an action with `type: "controls"`, `actor_id`, `goal`, and
`inputs`: one `{ "keys": ["w"], "mouse": [0, 0] }` entry per 1/24 second.
Keys: w/a/s/d, shift, space, c (toggle crouch), stand, attack (held staff).
Mouse deltas are pixels. This deterministic path shares terrain, body collision
and NPC policies with whitebox actions. It needs no per-tick model planning.
Unsupported prone/firearm actions fail explicitly. Use the same plan, compile,
video review and commit flow; recording inputs does not make them accepted history.
The trace must end grounded, outside a posture transition or staff sweep, and
have `len(inputs)+1 = 17*k+5`. Do not append a forced stop for frame alignment.
See `docs/keyboard_controls_and_memory.md` for the interactive preview.

# Paired Memory and Geometry Tools

Current renderer default: epoch-15 LoRA, controls=[] (no depth ControlNet).
Render args include `db` alongside `request` and `renderer` to retrieve paired
history. The tool chooses the nearest historical view to the planned endpoint
and saves memory_selection.json. Do not put media tags into action prose.

`memory_capture`: {db, output, initial_rgb?}. For revision-zero initialization,
initial_rgb is the original source image. Later capture uses only the last
accepted video's terminal RGB and the committed corrected scene. `commit`
automatically attempts capture; a reported capture failure can be retried.

`memory_query`: {db, camera?, limit?}. Inspect stored paired views and their
camera differences. Memory is evidence, not a new state update.

`geometry_propose`: RGB-only candidate reconstruction. See the returned
candidate and diagnostics; preview before initialize, never overwrite accepted
world geometry from a fresh reconstruction. No automatic world merge exists.

For image-based initialization, use geometry_propose and inspect its perception
results, masks and source candidates before revising a scene. Reuse matching-image
caches when available. Read docs/geometry_pipeline.md for the arguments. A successful
proposal is not an alignment approval: compare the original RGB with rendered
camera views, especially background landmarks, poles, road boundaries and actor
scale. Use detected boxes/masks as screen-space evidence, not exact 3D truth.
Preserve observations and write revisions to new scene files; record which object
correspondences support each correction and what remains uncertain. Inspect a new
preview after edits. Never convert a known inaccurate initialization into a trusted
generation-control baseline just because schema or collision checks pass.

