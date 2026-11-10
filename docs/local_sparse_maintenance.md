# Sparse observation maintenance diagnostic

This experiment keeps all static geometry fixed. It fits camera translation from
three annotated static anchors, then intersects actor foot rays with an explicitly
assumed ground plane. It generates candidate snapshots, not accepted world history.
It neither creates a continuous trajectory nor infers events between observations.

## Reproduction

From the project directory on Windows (SciPy installed):

```powershell
python scripts/align_sparse_observations.py --scene outputs/local_landmark_fit_20261109/urban/scene.json --annotations configs/urban_sparse_maintenance.json --output outputs/local_sparse_maintenance_new
python -m unittest scripts.test_observation_alignment scripts.test_image_alignment -q
```

Render each candidate through WSL with `python3 -m whitebox.godot.render`,
`--use-display`, `--width 1280 --height 720`, and a fresh output directory.
The current WSL Python lacks SciPy; the numerical solver ran on Windows and
the actual renderer ran in WSL. No API or neural inference was used.

## Real-frame evidence, November 9

Output: `outputs/local_sparse_maintenance_20261109/`.
`comparison.jpg` contains two actual source frames and two actual Godot renders.
Both render invocations completed with exit code zero using llvmpipe.

| Relative time | Numerical update time | Static-anchor residuals |
| --- | --- | --- |
| 4.532 s | 0.0093 s | 3.30, 2.18, 2.95 pixels |
| 9.056 s | 0.0048 s | 4.73, 2.50, 4.19 pixels |

Times exclude human annotation, source-image loading, and rendering. Annotations
are approximate Codex visual estimates, not dataset ground truth. The static
landmarks were initially fitted to the first image, so these are later-view
consistency observations, not independent proof of metric reconstruction.

Both steps preserve static entities exactly and carry actor IDs forward. The
second update begins with the first candidate. Source files and accepted worlds
are not modified. Unit tests independently recover known synthetic camera/actor
translations and verify input immutability and unchanged static entities.

## Visual findings

Actor ground contact locations follow the observed movement. NPC screen size and
player height are still inaccurate; fixed human height and ground/camera priors
do not jointly explain the image. Buildings and roadside object silhouettes
remain inaccurate even when ground contacts align. Initial camera/scale ambiguity
must be resolved against more evidence rather than fitting every object anew.

The current solver assumes fixed camera orientation and intrinsics. It must not
be presented as a general camera-motion estimator or used to accept a rotating
camera sequence merely because its optimization converged. Anchor residuals stay
in the report. Occlusion, body posture, support changes and contact events are
not checked. These candidate snapshots are not committed to a world database.

## Independent garage initialization

`outputs/local_garage_agent_20261109/source_whitebox.png` shows an RGB-only agent
construction with two actual renders. Geometry checks pass after one revision,
but the visual result fails: horizontal placement, framing and subject height
are wrong. This confirms the missing camera/image alignment is not specific to
the urban street. The coarse GTA human representation itself is intentional.

Next work: add independent vertical/silhouette constraints to camera-scale
calibration, evaluate more distinct scenes, then connect verified observation
updates and newly visible geometry to the existing explicit revision tools.
