# Garage temporal diagnostic

Date: 2026-11-10. Local computation only; no paid model call, no world commit.

## Question

Can the initial inferred garage remain fixed while its camera is aligned to the
actual video at encoded timestamp 1.0 seconds? This is a maintenance diagnostic,
not a rendering-quality benchmark.

Source video: sample
`ep_20260923_122317_11a9_agent_000_segment_00000_f000240/target_rgb.mp4`
under `E:/GTA_555_complete_rebuild_20261025/available_source_training/training/samples`.
Frame: `outputs/local_garage_sequence_20261110/frames/000024.png`.
The four fitting points and two independent rear-door checks are approximate
manual image annotations in `configs/garage_next_camera_landmarks.json`.
No dataset camera or ground-truth geometry was used for fitting.

## Result

| Candidate | Fitting-point errors (pixels) | Rear-door check errors (pixels) |
| --- | --- | --- |
| Initial camera | 45.8, 40.3, 72.9, 56.7 | 45.0, 62.3 |
| Free field of view | 11.9, 6.2, 14.6, 4.0 | 165.2, 137.2 |
| Fixed field of view | 8.8, 10.2, 16.1, 2.6 | 122.9, 85.8 |

Free fitting changed field of view from 40 to 32.14 degrees. Locking it at
40 degrees reduced, but did not solve, the independent-check error. The locked
fit also reached its allowed camera-height change of -1 metre. Neither candidate
passes visual maintenance review. Optimizer convergence is not world alignment.

The explicit visual evidence is
`outputs/local_garage_sequence_20261110/camera_evidence/comparison.jpg`:
circles mark image annotations, crosses mark projections; green points were
fitted and orange points were held out. All three panels were visually reviewed.
The entrance improves while the interior door becomes worse.

This rejects the assumption that camera-only fitting already maintains this
initial scene. It does not uniquely identify the bad parameter: inferred static
depth/size, approximate correspondences, and camera assumptions remain coupled.
Keeping field of view fixed here is a diagnostic hypothesis, not measured camera
metadata. Next calibration should use several frames with shared static geometry
and independent checks; moving the rear door separately for each frame would hide
the problem rather than maintain one world.

## Reusable changes

`whitebox/camera_fit.py` accepts equal lower/upper parameter bounds to fix chosen
parameters. Only free parameters enter the optimizer. The synthetic recovery
test covers a fixed nonzero field-of-view offset.

`scripts/visualize_camera_evidence.py` overlays any number of scene candidates
against fitting and held-out annotations on the actual source image. It rejects
image/annotation dimension mismatch and writes a JSON record without modifying
the scenes. Example:

```powershell
python scripts/visualize_camera_evidence.py --image outputs/local_garage_sequence_20261110/frames/000024.png --landmarks configs/garage_next_camera_landmarks.json --scene outputs/local_garage_visibility_fix_20261110/scene.json --scene outputs/local_garage_sequence_20261110/step_01_camera/scene.json --scene outputs/local_garage_sequence_20261110/step_01_fixed_fov/scene.json --output outputs/garage_camera_review
```

Separately, actor-root maintenance now translates existing pose points and held
staff endpoints together instead of recreating a canonical pose. This preserves
unobserved articulation while applying a root translation; it does not infer
new articulation from a foot observation.

## Shared-geometry follow-up

`scripts/fit_scene_landmarks.py` now accepts `views`, each with its fixed camera,
image dimensions, fitting landmarks and optional held-out landmarks. All views
use one set of entity parameters. The original single-image API is retained.
The CLI writes a shared scene plus per-view camera scenes; it never commits a
world. Synthetic two-camera tests recover a shared XYZ displacement, preserve
inputs, and verify that changing held-out pixels does not affect fitting.

Actual garage experiment:
`outputs/local_garage_sequence_20261110/shared_door_diagnostic`.
Using frames 0 and 24, both inferred cameras held fixed, fit only the rear
door's shared X/Z position. Height stays at its original 2 m prior. Ground-level
door-center annotations are fitting evidence; door-top annotations are held out.
The first-frame annotations are approximately (958,355) and (958,245).
These are manually inspected RGB points, not dataset ground truth.

The fit took 0.0094 seconds. Base errors became 55.1/84.9 pixels across the two
views, and held-out top errors were 58.0/73.3 pixels. The first frame got worse.
No geometry collision was reported, but this is still a failed alignment.
The candidate door moved from Z=6.25 to Z=9.31; this is not a validated repair
and was not adopted. The actual next-frame overlay is in
`shared_door_review/comparison.jpg` beside the experiment directory.

This test shows that simply giving each visible object more freedom does not
resolve inconsistent camera/geometry assumptions. The next step needs joint
multi-frame calibration with shared geometry, rather than independently fitting
camera and object against a few image points.

## Joint calibration and visibility failure

The fitter now accepts optional per-view `camera_bounds`: six delta ranges for
X/Y/Z translation, yaw, pitch, and vertical field of view. Equal endpoints lock
a parameter. At least one view must have a fixed camera. Camera changes and
shared entity parameters are optimized together; output `view_cameras` records
the resulting cameras. Existing single-view and fixed-camera callers still work.
The scene's original camera is retained in the shared geometry output; CLI
`view_00.json`, etc. combine the shared geometry with each fitted camera.

Synthetic tests independently cover geometry/camera recovery, fixed anchors,
held-out independence, unchanged inputs, backward compatibility and invalid
camera bounds. This API is still candidate fitting, not an acceptance mechanism.

Actual experiment: `outputs/local_garage_sequence_20261110/joint_calibration`.
It fixes the initial camera and both FOVs, shares door/bollard X/Z and jamb X
parameters, and fits the next camera's translation/yaw/pitch. Four foreground
anchors plus the door base are fitted in each frame. Door tops are held out.
All source annotations are approximate manual RGB annotations, not GT geometry.

Optimization took 0.052 seconds. Held-out door-top errors are 39.6 px at the
first frame and 6.3 px at the second. Left-jamb residuals remain 40.0/29.4 px.
The nonzero singular values and absence of active bounds describe this local
fit only; they do not establish correct world geometry or global uniqueness.

Actual Godot rendering completed locally using llvmpipe and was visually
inspected: `joint_calibration/render_01/camera.png`. The door is missing.
The fitter moved its center to Z=8.49 while the back wall stayed at Z=6.4.
The coarse visibility test on a manually selected door-interior image rectangle
finds 344/344 comparable rays blocked by `back_wall` (see `visibility.json`).
The rectangle is not a SAM result. Actor positions were intentionally untouched
in this static-scene calibration, so the render is not a completed action update.

Conclusion: joint reprojection can improve while the actual rendered scene gets
worse. Door/wall attachment and shared room extents need representation-level
constraints, and candidate review must include visibility as well as projection.
No candidate from this experiment was adopted or committed. Do not present the
6.3 px result by itself as a successful world-maintenance result.

## Explicit surface attachment

`whitebox/surface_attachment.py` projects a static box child onto a declared
parent-box face. The normal accounts for parent yaw and the child's oriented
support radius. Tangential coordinates and orientations remain unchanged.
The declaration is explicit, for example:

```json
{"child":"rear_door","parent":"back_wall","axis":2,"sign":-1,"gap":0.01}
```

`fit_scene_landmarks` applies `attachments` on every candidate. A child normal
position cannot also be an independent fit parameter, including through linked
translations. Tangential parameters remain available. Tests cover rotated faces,
child yaw, gap, unchanged input, invalid relations, and fitting integration.

This is a face-plane constraint, not a full wall/opening asset: it does not cut a
hole, fit a hinge, guarantee containment within the finite face, or infer which
surface an object belongs to. Chains are explicitly unsupported. Relations are
currently supplied by the fitting annotations, not automatically persisted as
world-memory facts or enforced by every world-edit tool.

Experiment `outputs/local_garage_sequence_20261110/attached_calibration` repeats
the joint fit with the door's depth derived from its wall face and its horizontal
position still fitted. It takes 0.0615 seconds. Held-out door-top errors are
34.9/22.3 px across frames; left-jamb errors are still 40.3/37.9 px.
The real Godot render at `render_01/camera.png` was inspected: the door is now
visible. On exactly the same manual image rectangle, 4/373 comparable rays are
blocked (by the blue container), versus 344/344 blocked by the back wall before.
The comparable ray count changes because the fitted proxy covers different
pixels; the image evidence rectangle itself did not change.

This fixes the wall-hiding failure, not the complete reconstruction. The greater
held-out projection error is preferable evidence to an invisible but numerically
aligned door. No result was committed, and actor motion remains out of scope for
this static calibration experiment.
