# Local whitebox diagnosis

The local workflow now renders through WSLg and Godot 4.5. No remote server or
paid model API was used. Subagents used astra low, one worker at a time alongside
the main agent. All scenes here are candidates in new output directories; no
accepted database or historical experiment was edited.

## Reproducible problems and fixes

1. A camera FOV edit was ineffective when intrinsics remained present: Godot
   prioritizes intrinsic focal lengths. Refinement and video reconciliation now
   update focal lengths consistently and preserve the calibration ratio and up.
   The local render diagnostic confirms the resulting image changes.
2. Previous camera fitting projected with global Y-up and scalar FOV even when
   the actual camera had explicit up/intrinsics. A shared image_alignment module
   now matches Godot. fit_initial_camera uses it and retains camera metadata.
   Six real renders, including an anisotropic calibrated rolled camera and its
   FOV revision, agree with numerical checkpoint projections within 0.001 pixel.
3. WSLg conflicts with private Xvfb Unix sockets on this installation. The proposal
   pipeline accepts the renderer's existing use_display option. It can now compile
   and render locally without changing shared X server directories.
4. A prior worker fixed duplicate observation IDs affecting dynamic occlusion
   masks during ground completion. See local_geometry_diagnostic.md for the
   independent failing regression and fix.

## Real image corrections

`scripts/fit_scene_landmarks.py` takes explicit source-image landmarks and a small
set of editable transform axes with bounds. Linked pieces share translation,
allowing a pole and its base or a tree and crown to move together. Camera and all
unselected geometry stay fixed. The output is an uncommitted candidate plus a
source-image overlay and error report. Geometry errors are reported explicitly.

Source image annotations were made visually by Codex. No dataset geometry was
used to obtain their pixels. These are approximate annotations, not detector
measurements or true 3D positions.

| Case | Landmark errors before | Fitting time | Actual before/after render times |
| --- | --- | --- | --- |
| Urban: lamp base, hydrant, sign post | 58.1, 45.5, 35.2 px | 0.062 s | 7.48 / 7.49 s |
| Rural: near pole and two palm roots | 197.5, 69.8, 108.4 px | 0.017 s | 5.18 / 5.46 s |

All fitted anchor errors in the rendered images are below 0.001 pixel, but each
anchor supplies two equations for two position coordinates. This is fit residual,
not held-out accuracy. It proves the correction command reaches the renderer.
The rural sheet exposes its limit: roots align while crown height and shape still
do not. Ground-plane and camera priors determine inferred depth. Buildings and
terrain outside the annotation set are unchanged. Neither scene is declared
geometrically correct or ready for a long orbit.

Main review artifacts, relative to agentic_world:

- outputs/local_alignment_render_20261109/urban_methods.jpg
- outputs/local_alignment_render_20261109/rural_comparison.jpg
- outputs/local_alignment_render_20261109/urban_comparison.jpg
- outputs/local_alignment_render_20261109/report.json
- outputs/local_landmark_fit_20261109/{urban,rural}/report.json
- outputs/local_landmark_fit_20261109/{urban,rural}/overlay.jpg

## Cached geometry path

Unlike the rural download, the urban archive includes reconstruction NPZ and
perception masks. The current compiler processed those locally, retaining source
RGB correspondence checks, and rendered the result. The original inference
remains cached; no fresh VGGT or SAM inference is claimed.

Command from the project directory inside WSL:

```bash
python3 -m whitebox.geometry_pipeline --args configs/local_urban_geometry_cache.json
```

The configured output is immutable per run; choose a new output for repetitions.
Measured wall times in proposal.json: compile 2.139 s, preview 3.682 s, total
6.339 s. Reconstruction and perception are explicitly marked cached with zero
new inference time. Visual inspection shows oversized tree volumes, disconnected
facades and ground holes; successful compilation does not resolve those issues.

## Verification and next work

```text
python -m unittest scripts.test_image_alignment scripts.test_geometry_pipeline scripts.test_local_camera_update scripts.test_camera_reconcile scripts.test_geometry_ground scripts.test_local_world_maintenance -q
Ran 76 tests in 7.214s; OK
```

Some maintenance test classes inherit existing tests. This is an execution count,
not 76 independent scenarios. The six real renders ran separately in WSL through
scripts/render_alignment_diagnostic.py and its report records their timings.

Remaining scope is substantial: hold-out silhouette/top landmarks, real sequential
observation updates, revisited-view consistency, inferred ground completion,
background extension with retained identities, and end-to-end initialization
latency with fresh local inference where practical. The current fixed-camera
landmark tool accelerates corrections but does not complete that workflow.
