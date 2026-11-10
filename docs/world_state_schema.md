# World State Inspection

## October 9 Whitebox Extension

The independent Godot whitebox path reuses the existing tables without migration.
`objects.attributes.whitebox` stores resource parameters, articulated pose,
weapon endpoints and NPC behavior/progress; entity position/rotation remain in
the existing canonical columns. Godot [x,y,z] maps to canonical [x,-z,y].
`world_meta.whitebox_header` holds terrain, world lore and engine-camera settings;
`whitebox_entity_order`, `whitebox_revision`, `whitebox_history`, and `tick` retain
execution ordering and accepted history. Exported scene/timeline files are derived
candidate snapshots, not a second authoritative state. Appearance reference
metadata lives under `appearance_reference_memory`; images are separate assets,
not independent evidence. `appearance_captures` records which accepted video
revision has been indexed, with reference IDs and extraction uncertainties.
These keys are derived memory metadata; indexing does not advance `tick`, edit
objects, or rewrite `whitebox_history`. Reference evidence includes the accepted
video, frame ID, video timestamp and revision. See
`appearance_memory_20261012.md` for the active retrieval and native H3 path.
The new path does not run the old cuboid H3 compiler.
See `whitebox_20261009.md` for tested scope. The H3/environment findings below
describe the earlier October 3 inspection and are not the current Comfy inventory.

Inspected on server star-Super-Server, 2026-10-03 (Asia/Shanghai).

## New Episode Payload

The subsequently supplied `ep_20260812_090725_f8a5.zip` DOES contain actual
wire-schema 0.1.0 world payloads: 17,540 JSONL/Parquet snapshots, camera and
aligned frame timestamp payloads, seven entity lifecycle rows and recorded MKVs.
The prior missing-payload finding below applies only to the earlier archives.
This episode is explicitly partial and its quality report is invalid; it is not
a complete static map. The first snapshot passes the existing GTA adapter.
See `episode_payload_and_comfyui.md` for schema fields, measured counts,
server validation, camera/time alignment limitations and reproducible commands.

## Existing Representation: Server Search

Initially no project world-state SQLite database or implemented white-model representation
was found under `/home/star/minimax-h3`, `/mnt/hdd/DiffSynth-Studio`, or in
the top five levels of `/home/star` and `/mnt/hdd`. Discovered desktop index,
editor/Codex state and unrelated experiment-analysis databases are not world
databases and are not opened or modified. This is a bounded search, not a
claim about every file on the machine.

`/home/star/minimax-h3/deploy/WORLD_STATE_RENDERING_OVERVIEW.md` is a proposal:
it names entities, transforms, camera and uncertainty but defines no persisted
schema. Lingbot's installed generative world pipeline is not an authoritative
entity store. Consequently there is no existing object ID convention, camera
calibration, relation table or temporal entity state to adapt here.

## Supplied GTA Archive (Subsequent Inspection)

The user subsequently supplied `GTA_0906 (2).zip` and `video_to_whitebox.zip`.
This supersedes the initial server-only conclusion: there IS an existing canonical
white-model schema in GTA source, though no recorded entity payload is included
in the supplied GTA archive. Full `.tables`/`.schema` equivalents, counts and
sample rows for all three databases are in `gta_database_inspection.json`.
Databases were opened read-only, including the supplied controller WAL.

| Database / table | Count inspected | Meaning / authority |
| --- | ---: | --- |
| controller / episodes | 97 | Recording lifecycle/configuration, not current world entities |
| controller / client_assignments | 92 | Episode/client to agent assignment |
| controller / connector_sessions | 113 | Client connection registry |
| controller / heartbeats | 88 | Capture/encoder health metadata |
| ingestor-derivations / jobs | 1 | Proxy derivation queue/cache state |
| ingestor / accepted_records | 1,073,401 | Deduplication keys ONLY: episode/channel/producer/sample |
| ingestor / world_clock_domains | 22 | Named clock-domain metadata |
| ingestor / world_clocks | 40 | Last world sample sequence and receipt/server times |

None is an entity/world fact table. Do not invent positions from accepted_records.
`src/wm_engine/ingestor/store.py` writes actual payload to
`<episode>/.spool/world_state.jsonl`; finalized datasets use Parquet payload_json.
No `.jsonl`, `.parquet`, `static.sqlite3`, or recording `rgb.mp4` was found in the
GTA archive. The inverse archive contains original sampled JPEG frames and
rendered videos, with manifests referencing external source RGB paths.

Reuse `src/wm_engine/core/schema.py` canonical 1.0.0 and
`dataset/legacy.py` wire 0.1.0 conversion. Entity ID is
`session_id:dataset_entity_id`, not a network ID/local handle.
EntityState has position_xyz, optional quaternion xyzw, native Euler plus explicit
rotation convention, scale_xyz, linear/angular velocity, owner, health, state,
spawn/despawn ticks and times. WorldSnapshot completeness is explicitly partial.
CameraState has optional view/projection matrices, FOV axis, clips, resolution.
Missing calibration MUST remain missing; native ROT_ZXY degrees cannot silently
be read as XYZ radians. Coordinates are GTA +X east, +Y north, +Z up,
nominal world-unit metres, not physically calibrated.
Categories are entity_type/model, not full visual semantics.
Relations exist as owner/current_vehicle/seat and state, not a relation table.
Measured recorded payloads are original evidence; proxies are derived render hints.
No existing database migration is needed: keep inputs read-only and store only
the selected, ID-preserving episode working state and explicit edits separately.

Inverse `scene.json` v1/v2 has camera position/target keyframes, perspective or
orthographic projection, objects with shape and position/rotation/scale keyframes,
and characters with 15-joint keyframes. There is no reliable semantic category for
generic geometry. Its own assumptions explicitly describe monocular approximate
depth, dimensions and camera trajectory. Imported entries use INFERRED provenance,
unset confidence rather than fabricated confidence, preserve assumptions/source
IDs, and never count as independent ground-truth evidence.

## Prototype Working-State Schema

The prototype creates its own small `data/world.sqlite`, never modifying H3.
Full SQLite table/schema output is produced by `scripts/inspect_world.py`.

| Table | Purpose | Authority |
| --- | --- | --- |
| objects | Stable object_id, category, position, rotation, scale, bbox_3d, attributes, state, confidence, provenance | Current authoritative world |
| world_meta | Camera intrinsics, world-to-camera matrix, scene intent, tick | Authoritative scene configuration |
| edits | Before/after object JSON, reason, evidence IDs, timestamp | Audit history, not another world |
| evidence | Explicit original input/manual evidence records | External evidence; never generated observations |

JSON fields hold vectors and dictionaries; rotations are XYZ Euler radians,
positions/scales are metres, camera uses +X right, +Y down, +Z forward.
`bbox_3d` is local XYZ extent; the compiler rotates its eight corners.
`state` can carry visibility/velocity or other temporal state; tick and edits
record updates but this is not a dynamics simulator. Relations can be carried
as attributes in this version; there is no dedicated relation graph.
IDs are caller-assigned stable strings, independent from observation track IDs.
Allowed provenance: OBSERVED, INFERRED, AGENT_CREATED, RENDER_ONLY, UNKNOWN.
RENDER_ONLY objects are excluded from authoritative expected-object queries.
Appearance embeddings in objects are reference matching metadata, not facts.
No renderer output is automatically promoted to evidence or world facts.

`data/observations.sqlite` is separate:

| Table | Purpose |
| --- | --- |
| frames | Episode, attempt, frame number, time, file |
| observations | Frame-local observation ID, episode-scoped track ID, bbox/mask/embedding/depth and source |
| reports | Attempt-level matching/discrepancy JSON |
| decisions | Agent decision, reason, requested edits and executed outcome |
| artifacts | Observation mask path and reference embedding vector |

Masks/embeddings/video live in episode directories, not database blobs.
Conversational memory is a bounded list of recent decisions plus current goal;
world objects are fetched by tools and are not appended wholesale to history.

## H3 Contract

Source: `diffsynth/pipelines/minimax_h3_audio_video.py`, `MiniMaxH3Pipeline.__call__`.
Deployed full INT8 recipe: `MiniMax-H3-Int8-ConvRot-Ref2VA.py`.
Inputs: prompt, height, width, num_frames, num_inference_steps, seed, cfg_scale,
references (image / video / audio / video_audio). Returns `(video, audio)`.
Video reference lists must already be 24 fps. Internal frames align to 17n+5;
the wrapper trims delivered frames/audio to the requested duration.

The common pipeline signature also contains keyframes and control_video, but
keyframes require the FL2VA recipe and control_video requires an additional
ControlNet checkpoint. No deployed ControlNet weight was found.
Ref2VA `first_frame` in our wrapper means **image reference**, not locked frame 0.
Depth, semantic IDs and camera are compiled/debug artifacts, not supported H3
conditioning channels in this installed recipe. Explicit unsupported condition
arguments raise errors; nothing is silently fed to an unrelated input.

## Environment

Existing Python 3.11.15, PyTorch 2.9.1+cu128, CUDA runtime 12.8;
driver 580.178.04, four RTX 4090 devices reporting 49140 MiB each.
H3 uses existing `/home/star/minimax-h3/.venv`, GPU 2 (encoder/VAE) and
GPU 3 (DiT), existing INT8 weights on `/mnt/hdd/MiniMax-H3-Quant`.
Perception is run after the renderer child exits to release its allocations.

## Coarse Posture Extension (2026-10-10)

The whitebox adapter additionally stores optional actor `posture_weights`,
`support_contacts`, and `grip` inside the existing `objects.attributes.whitebox`
JSON. No table or column migration is required. Accepted coarse pose parameters
are persistent episode state; `pose` joints are their procedural realization,
not a recovered video skeleton. Support contacts are derived template geometry.
The staff's existing `holder` remains the ownership relation, independent of
one/two-handed grip. Pending predicted posture is not committed until video
acceptance and reconciliation. See `posture_reconcile.md` for the vocabulary,
provenance, unsupported changes and real-video trial.

