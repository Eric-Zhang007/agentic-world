# Camera calibration and image-backed extension

## Camera-only correction is insufficient

Run from the project directory on Windows:

```powershell
python scripts/fit_initial_camera.py --scene outputs/local_landmark_fit_20261109/urban/scene.json --landmarks configs/urban_camera_landmarks.json --output outputs/new_camera_fit
```

The new landmark format supports camera translation, yaw, pitch and FOV fitting
against fixed geometry. Explicit six-parameter bounds replace the old script's
scene-specific camera ranges for this format. Existing joint-annotation callers
still use the old path. Projection follows Godot's up vector and intrinsics.
Optional held-out points are reported separately and do not enter optimization.

Actual November 9 run: `outputs/local_camera_calibration_20261109/urban/`.
The numerical fit took 0.0225 seconds, 12 evaluations. Main actor top error fell
from 34.86 to 1.81 pixels, with all five fitted anchors below two pixels.
But held-out NPC root/top errors increased from 27.49/40.12 to 49.31/65.90 pixels.
An actual WSL Godot render completed successfully, and `comparison.jpg` was
visually inspected. The player scale improves while NPC and background geometry
remain wrong. This candidate is not an accepted improvement of the whole scene.

The landmarks are approximate visual annotations. Actor top is an upright-height
prior, not a recovered head joint. In particular, a running person does not have
the standing silhouette assumed by this simple calibration. The test demonstrates
the need to distinguish camera, shape and posture errors; it does not establish a
correct metric camera. No assets were resized or moved during this camera fit.

## Add a missing image primitive without moving the world

```powershell
python scripts/propose_image_primitive.py --scene outputs/local_landmark_fit_20261109/urban/scene.json --annotation configs/urban_bollard_annotation.json --output outputs/new_image_extension
```

This composes existing ray/plane intersection, landmark fitting, and
`scene_refinement.apply_proposal`. The result is a normal evidence-backed
INFERRED addition, not an alternate world schema. It is not committed.
Annotation specifies a base pixel, support-height prior, dimensions prior,
landmarks, and the dimensions/transforms allowed to change.

Actual run: `outputs/local_image_extension_20261109/urban/`.
The missing far-left yellow parking bollard is lifted from the original urban
RGB. Total computation was 0.0112 seconds, excluding annotation and rendering.
Its base error is effectively zero; top error is 1.53 pixels. Inferred height is
1.393 m under the supplied camera and ground prior, not a physical measurement.
The thickness remains an explicit 0.16 m prior. A real Godot render shows the
new bollard at the annotated location. Existing entities and camera are exactly
unchanged. Background remains crude, and occlusion is not independently checked.

Outputs include `scene.json`, `proposal.json`, `report.json` and actual render
images. The proposal can be reviewed by existing refinement tooling. Historical
and joint visual validation are still required before any authoritative commit.

## Persistence and scope

`scripts/test_local_extension_history.py` separately tests existing commit tools
against real temporary SQLite databases containing accepted synthetic history.
Four tests pass: additions preserve previous state/history, rejected conflicts
are atomic, and creative extension cannot be used to transform existing objects.
See `docs/local_extension_history.md` for exact coverage.

No existing accepted world, archived timeline, remote server or model API was
used or changed. These runs establish local candidate construction and database
invariants, not end-to-end autonomous scene recovery or a completed video loop.
Multi-view camera/geometry calibration, real observed-history extension, and
fresh perception latency remain open work.

## Preserving explicit support during fitting

On November 10, the image-primitive path was changed to preserve its declared
horizontal support plane throughout optimization, not just during initial ray
lifting. `fit_scene_landmarks` accepts `support_heights: {box_id: height}` and
derives center Y from `height + sizeY/2` at each evaluation. Explicit center-Y
parameters, including linked translations, conflict with this constraint and are
rejected. Surface attachment and a support-height declaration on the same child
are also rejected rather than silently overriding one another.

This supports upright/Y-rotated static boxes, not arbitrary inclined contact or
physics. The plane is a supplied prior; preserving it does not establish that it
is physically correct. Existing no-support callers keep their prior behavior.

Re-run of the original actual urban bollard annotation:
`outputs/local_supported_extension_20261110`. Total numeric proposal time was
0.0108 s. The bottom remains at the declared Y=0.045, top image error is 1.53 px,
and inferred height is 1.393 world units. A fresh local Godot render was inspected
at `render/camera.png`. All preexisting entities and camera remain unchanged.
The former unconstrained fit happened to satisfy the plane on this particular
example too; this run demonstrates preserved output, not a measured visual gain.
No world or accepted history was committed.

For new `propose_image_primitive` calls, fit size Y rather than center Y. The
center is now derived from the explicit support. This avoids silently changing
the declared support when dimensions or inconsistent image anchors change.
