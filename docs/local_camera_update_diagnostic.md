# Local camera update diagnostic

## Finding

Godot render.gd set_camera prioritizes intrinsics over fov: vertical FOV is
2 atan(height / (2 fy)). Updating only the stored fov therefore changes the
camera dictionary without changing projection. The regression reproduces this
with fixed intrinsics and a ten-degree FOV change. This is a projection-level
reproduction, not a rendered pixel comparison.

## Update contract

- Refinement fov is an absolute vertical projection angle.
- Reconciliation fov_offset is relative to the planned camera's effective
  projection angle, not its potentially stale fov metadata.
- Both use update_camera. With intrinsics, FOV edits scale fx and fy together,
  retaining focal aspect ratio, principal point, image dimensions and metadata.
- Position and target edits preserve intrinsics and explicit up. Up is a
  look-at reference axis, not a cached rotation. Parallel explicit up is rejected;
  absent up uses the renderer's near-vertical fallback, allowing overhead views.
- Zero FOV offsets do not rewrite calibration or stale metadata.
- Helpers copy input cameras. Refinement changes only the candidate;
  reconciliation validates copies before replacing corrected frame cameras.
  Existing archival commit semantics remain unchanged.

No renderer files, server calls, model inference or agents were used.

## Verification

Run python -m unittest scripts.test_local_camera_update
scripts.test_camera_reconcile scripts.test_scene_refinement.
Coverage includes the legacy precedence reproduction, consistent absolute and
relative updates, retained calibration shape, unchanged inputs, zero offsets,
explicit up degeneracy, overhead views and existing archive/commit regressions.
