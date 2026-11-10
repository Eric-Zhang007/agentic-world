# GTA asset alignment

The resource geometry now comes from the frozen dataset renderer under
`E:/GTA_555_complete_rebuild_20261025/source/method/source/src/wm_engine`,
instead of approximating its appearance from screenshots.

## Integrated resources

- `assets/gta/catalog.json`: the original 13 rigid human parts, including the
  source arm spread and face direction cues. The renderer no longer renders
  procedural joint poses. Entity position, yaw and overall dimensions remain
  controllable. Coarse posture now scales the whole body height using the shared
  profile; individual joints are not animated. See `semantic_actions.md`.
- `gta_vehicle`: original direction-aware wedge, selected with `asset_id` on a
  `box` entity. Box origins remain centered; actor origins remain foot-based.
- `scene.static_geometry.path`: regional classified collision surfaces,
  vegetation and props exported using the dataset's `scene_geometry` and
  `surface_colors(..., 'relief')`. This replaces the procedural ground during
  rendering. Building bounds are not converted to solid volumes.

`scripts/export_gta_asset_catalog.py` regenerates the catalog from source.
`scripts/export_gta_region.py` exports a regional binary XYZ/RGB triangle mesh
and its source metadata. Both read the existing dataset; no OSS access is used.
Relative static mesh paths resolve relative to the scene JSON. When moving a
scene, move its referenced mesh too or supply an absolute destination path.

## Actual verification

The urban region centered at GTA coordinates
`[-1187.82495117188, -1525.2451171875, 5.11748790740967]`, radius 100 m,
exports 185020 triangles. It was rendered with the existing server Godot
renderer in four views. Road, sidewalks, poles and building surfaces are visible.
The actor and vehicle catalog also passed a four-view render.

Local results: `outputs/gta_asset_alignment/region_preview.png`,
`catalog_preview.png`, `rigid_human_camera.png`.
Server results: `/home/star/agentic_world/outputs/gta_asset_alignment/`.

The earlier `region_preview.png` used provisional 60-degree vertical FOV.
`calibrated_camera.png` now uses the original frame's interpolated screen rays:
fx=856.80895, fy=770.93476 at 1280x720, vertical FOV=50.06191 degrees.
The original capture was 3072x1920, so the resized training frame needs distinct
horizontal and vertical focal scales. Godot renders at the calibrated aspect
ratio and resizes to the requested output. Fifteen 3D projection probes have
maximum error 0.061 pixels at 960x540. This is camera consistency, not a claim
that the RGB image and proxy contain identical geometry or appearance.

`camera_comparison.jpg` shows RGB, original whitebox, the earlier camera and
the calibrated static reconstruction. Dynamic actors are omitted in the latter.

Motion now queries `static_geometry.collision_path`, a separate export of the
original non-procedural collision triangles. Derived foliage boxes and visual
water are not automatically used as physical support. Walking uses bounded
vertical support queries, a 50-degree maximum slope, 0.35m maximum step,
0.5m maximum drop and triangle/body-box clearance. Body sweeps are sampled at
at most 1cm spacing. It does not calculate forces or find routes automatically.
Support queries retain the current height layer instead of snapping to a roof.
For Pi preview/initialize, relative resource paths are resolved against the
input scene before snapshots are written. Direct Python simulation callers
should provide absolute resource paths.

Ten actual street routes were tested. Nine reached their requested endpoints;
one requested 8m left but hit a static surface after 4.833m. It was reported as
an invalid plan and stopped, with zero terminal velocity. The motion report
retains all requested/actual paths. `surface_motion_paths.png` plots paths and
ground heights. `collision_walk.mp4` is a six-second whitebox-only diagnostic,
not an H3 generation or an accepted world update.

The dataset renderer can use observed body parts when measurements exist.
This catalog intentionally uses its rigid fallback, not inferred joint motion.
The prototype's lighting and vehicle edge styling differ from the original
renderer; geometry reuse alone does not establish identical conditioning.

Resources and camera rules are described in `configs/whitebox_resources.json`
and `harness/pi/world_instructions.md`. Agent planning receives collision
failure feedback; automatic obstacle detours remain future work. Historical
accepted scenes and videos were not rewritten.

Verification: 63 geometry/motion/reconciliation/world tests and two Pi tool
tests passed. Thin-wall rejection, low-step traversal, bridge-layer selection
and cliff rejection are covered. The server also rendered calibrated static
views and all 144 frames of the blocked-route whitebox diagnostic.
