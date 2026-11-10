# Input Data Assessment

Inspection date: 2026-10-03 (Asia/Shanghai)

This assessment covers the locally unpacked inputs:

- `input_data/GTA_0906`
- `input_data/video_to_whitebox`

Archive-level recheck on 2026-10-03 confirmed the distinction: GTA_0906 has
no world-state JSONL/Parquet payload files, whereas video_to_whitebox DOES have
reconstructed scene payloads (for example `test_data/GTA_fps5/segment_00001/scene.json`).
Those scene payloads are usable INFERRED world descriptions, not missing data;
they are not raw GTA entity-state recordings. The later episode ZIP supplies
the latter. See `episode_payload_and_comfyui.md`.

The inputs were inspected read-only. No source archive, SQLite database, or
existing application file was modified.

## GTA_0906

### Canonical schema

The authoritative source schema is
`src/wm_engine/core/schema.py`, version `1.0.0`.

- `Timestamp` records session/client/tick/clock-domain metadata.
- `SessionMetadata` describes an episode and its capture configuration.
- `CameraState` can contain position, quaternion or native Euler rotation,
  FOV, clipping planes, image resolution, and explicit view/projection
  matrices.
- `EntityState` contains a stable `capture_entity_id`, entity type, optional
  model identity, position, orientation, scale, velocity, owner, health,
  lifecycle ticks, and arbitrary state.
- `WorldSnapshot` is a timestamped list of entities and has an explicit
  completeness value.
- `Observation` combines camera, agent/input state, frame references, and
  body poses.

The legacy wire schema is `0.1.0` in
`src/wm_engine/dataset/legacy.py`. Its adapter preserves the original payload
and converts legacy IDs to `session_id:dataset_entity_id`. It explicitly does
not guess camera transforms or silently reinterpret native rotation
conventions.

### SQLite databases

The supplied archive contains runtime/control databases, not a self-contained
authoritative entity table:

| Database | Tables | Assessment |
| --- | --- | --- |
| `data/controller.sqlite3` | `episodes`, `client_assignments`, `connector_sessions`, `heartbeats` | Episode lifecycle, assignment, connection, and health metadata |
| `data/ingestor.sqlite3` | `accepted_records`, `world_clock_domains`, `world_clocks` | Deduplication keys and latest clock metadata |
| `data/ingestor-derivations.sqlite3` | `jobs` | Derivation queue state |

Observed row counts are 97 episodes, 92 client assignments, 113 connector
sessions, 88 heartbeats, 1,073,401 accepted-record keys, 22 clock domains,
40 world clocks, and one derivation job. `accepted_records` contains only
episode/channel/producer/sample sequence keys. It is not safe to reconstruct
world entities from it. Actual record payloads are expected in episode spool
JSONL or finalized dataset artifacts, which are not included as complete
payload stores in this archive.

### GTA data quality

The representation is structurally useful for an authoritative world state:
stable IDs, explicit timestamps, entity lifecycle, transforms, and optional
camera calibration are present in the schema. The supplied archive is not
complete enough to validate entity-level recall or temporal continuity because
the payload artifacts are absent. Treat `WorldSnapshot.completeness` and
missing optional fields as first-class uncertainty rather than filling them.

## video_to_whitebox

### Scene schema

The scene format is a separate JSON format with:

- `version`
- free-text `description`
- `assumptions`
- a camera with projection, FOV/orthographic scale, and position/target
  keyframes
- generic `objects` with `id`, `shape`, and position/rotation/scale keyframes
- `characters` with `id`, radius, and 15-joint keyframes

Representative files inspected include:

- `test_data/GTA_fps5/segment_00001/scene.schema.json`
- `test_data/GTA_fps5/segment_00001/scene.json`
- `test_data/GTA/segment_00001/scene.json`
- `outputs/METADATA_world/102-1/manifest.json` (no scene.json at this level)
- the GTA sample `manifest.json` files

The GTA FPS5 representative scene contains 13 generic objects, 2 character
tracks, 31 camera keyframes, and 9 assumptions. The older GTA representative
contains 13 generic objects, 2 character tracks, 17 camera keyframes, and 7
assumptions. Manifests contain source media, duration/FPS/resolution, and
sampled frame paths/timestamps.

### Important schema/data mismatches

1. The checked schema declares `version` enum `[1]`, while the GTA FPS5 scene
   is `version: 2`.
2. The checked schema allows shapes `box`, `sphere`, and `cylinder`, while the
   GTA FPS5 scene uses `rounded_box`.
3. Scene objects have geometry proxy names and colors, but no reliable visual
   semantic category, model identity, or appearance embedding.
4. The assumptions state that monocular depth, object dimensions, and camera
   trajectory are approximate. They are not measurement-grade calibration.
5. Camera keyframes provide position and target, but not captured camera
   matrices or a verified coordinate/calibration contract.
6. The scene format has no per-object confidence field. A confidence value
   must therefore remain `null`.
7. Generic object geometry is not equivalent to a detected object. It should
   not be promoted to an authoritative fact without external evidence.

The supplied reverse-reconstructed data is therefore useful as an inferred
rollout condition and for debugging spatial consistency, but it should not be
treated as ground truth. The importer in
`scripts/import_whitebox.py` preserves assumptions and source payloads,
labels imported entries `INFERRED`, and does not fabricate semantic labels,
depth, embeddings, or confidence.

## Import contract

`scripts/import_whitebox.py` imports one scene time point into:

- `WorldObject` records for generic objects and character tracks
- a local `Camera` record containing the selected pose and derived
  world-to-camera matrix

The selected timestamp must lie within every multi-key interpolated track's
time range. Single-key tracks are held static, as in the source renderer.
Rotations follow the source renderer's shortest-angle interpolation in degrees
and are exported as XYZ radians. The camera basis follows the source renderer's
Z-up convention (Y-up fallback for nearly vertical views); it is a derived
rendering transform, not measured calibration. All categories, including skeletal
character proxies, remain `unknown`; their source groups remain in attributes.
The JSON snapshot includes the source path, scene version, assumptions, and
raw provenance metadata so later reconciliation can distinguish these
inferences from captured GTA observations.

Run from the workspace root:

```text
python -B agentic_world/scripts/import_whitebox.py input_data/video_to_whitebox/test_data/GTA_fps5/segment_00001/scene.json --timestamp 1.0 --output snapshot.json
```

Requires the existing NumPy installation and the local WorldObject dataclass.
No H3 execution, database update, or dependency installation is performed.
