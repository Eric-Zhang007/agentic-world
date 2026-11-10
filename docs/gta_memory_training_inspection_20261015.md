# GTA memory training data inspection (2026-10-15)

## Scope and evidence

Read-only inspection of the partially downloaded `E:/GTA_555` dataset.
No training has run. Dataset-level completeness is not established.
Inspection script: `scripts/inspect_gta_memory_data.py`.
Artifacts: `outputs/gta_memory_inspection_20261015/`.

The inspected `0921` episode is `ep_20260921_124218_450d`.
Agent `agent_000`, segment `segment_00001` contains a 1440-frame RGB/proxy
pair at declared 24 fps. The frame metadata has 1274 estimated-alignment
rows and 166 missing-camera rows. The paired image artifact samples decode
indices 0 and 120, about five seconds apart. Four dynamic candidate IDs occur
in both samples; this does NOT establish four visible, appearance-consistent
reference matches.

Both sampled frames successfully join to the two bracketing camera records
by client sample sequence AND clock domain, and to their world capture tick.
This verifies record linkage, not geometric alignment. Camera payloads contain
rendered_camera_position, rendered_camera_rotation, rendered_camera_fov,
near_clip, far_clip, render_resolution and projection_rays (screen UV/world
point/direction). Camera and agent-state Parquet each have 129439 records;
entities has 18390, world state 90295, events 36699.
Local inspection installed pyarrow 23.0.1 from the Aliyun PyPI mirror after
the default download stalled. No remote inference environment was changed.

## Available representation

- `agents/<agent>/camera.parquet`: camera records (see inspection JSON for schema).
- `server/entities.parquet`: entity registry.
- `server/world_state.parquet`: temporal world state.
- `server/events.parquet`: events.
- `derived/segments/<agent>/<segment>/frames.jsonl`: RGB decode index,
  original and encoded PTS, proxy index, server-clock estimate/domain, world
  capture tick, client sample sequence, camera interpolation provenance,
  candidate entity IDs and scene index.
- `scene_*.json`: regional static placement IDs and collision surface
  mesh/component IDs, bounds and derived ground/obstacle classification.
- `appearance.json`: observed agent appearance intervals.
- `appearance_catalog.json`: model/appearance interpretation. In inspected
  0916 data, character_id explicitly means model number, NOT player identity.
- The 0916 world manifest reports identity_verified=false. Static identity
  across bundles requires validation, even when world names match.

The compiler documents rigid humanoids with fixed parts, not measured full
articulation. Scene candidate lists and semantic pixel totals are not
per-instance visibility masks. Some animals, litter and puddles are excluded.
The sampled whitebox exposes occupants through vehicle wireframes, whereas
RGB is dominated by opaque car body. Do not label those occupants as visible
solely because they were rendered in the proxy.

## Proposed sample contract (not yet a training loader)

Maintain three separate records:

1. Entity identity: episode-scoped dynamic ID, or verified world/version plus
   static placement/surface identity. Model ID alone is insufficient. Account
   for entity lifecycle and recycled runtime IDs.
2. Observation/state revision: capture time and clock domain, state revision,
   appearance coverage, camera/view direction, RGB/proxy frame addresses,
   world-state query address, bbox/mask and actual visibility evidence.
3. Reference-use edge: consuming sample/clip and entity, source observation,
   observed_at, used_at, applicable state revision, selected view, and reason.
   Preserve previous observations and edges when a new revision is added.

For a target clip [t0, t1], references must be available before t0. A reference
cannot be selected from later target frames. If the intended task allows a
first-frame condition at t0, mark that as a separate condition. Split by
episode/route/world region as appropriate before selecting references to
avoid near-duplicate train/evaluation leakage.

Model conditions should contain selected historical RGB+proxy frame pairs,
the target proxy sequence, and an explicit text mapping between entity,
reference index, visible region/view and applicable state. Target RGB is the
supervised output, NOT part of the historical reference input. Entity IDs
also require spatial grounding (mask/bbox or an ID map); an arbitrary number
in text alone does not locate a building or road patch in a crowded image.

Treat stable appearance separately from changing pose, damage, held objects,
doors and other state. Same identity does not require identical pixels under
different lighting/viewpoints. Old evidence remains historical; it is not
automatically valid for all future states. Ground references need local
surface patches and view coverage rather than a single whole-map ground ID.

## Next acceptance checks

1. Join frame metadata to exact/bracketing camera samples and world records;
   validate projection/timestamp agreement on RGB/proxy overlays. Exclude
   missing-camera intervals. Avoid using ordinal frame/24 as the sole clock.
2. Generate per-instance visibility evidence with occlusion handling, then
   validate a few dynamic entities, buildings and surface patches manually.
   Retain unknown labels instead of asserting visibility from scope lists.
3. Build a short reference-revisit sample and a state-change sample, keeping
   reference-use edges and state revisions inspectable. Only then connect
   the verified sample format to the actual H3 fine-tuning implementation.

Reproduce inspection:

```powershell
python agentic_world/scripts/inspect_gta_memory_data.py --episode E:/GTA_555/0921 --output agentic_world/outputs/gta_memory_inspection_20261015
```
