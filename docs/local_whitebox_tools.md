# Local whitebox tool entry points

Use these tools from the current Codex shell. They require no second model API.
Choose a new output directory for each experiment. Keep original source images
and prior accepted history unchanged.

| Need | Tool | Current evidence |
| --- | --- | --- |
| Coarse relative depth | `scripts/infer_local_depth.py` | Fresh GPU forwards on urban, rural, garage |
| Segment selected regions | `scripts/segment_local_boxes.py` | Fresh SAM2 masks on urban and garage; boxes/categories supplied by agent |
| Inspect source/proxy mismatch | `scripts/diagnose_mask_alignment.py` | Mask bounds versus coarse projected bounds; explicit associations |
| Review several scenes | `scripts/run_local_scene_review.py` | Urban/rural/garage with explicit coverage gaps; no implicit acceptance |
| Compare foreground order | `scripts/diagnose_depth_order.py` | Urban diagnostic; arbitrary depth units, no metric conversion |
| Track existing anchors | `scripts/compare_anchor_sampling.py` | Actual dense urban/rural videos; optional camera-only candidate |
| Fit selected static transforms | `scripts/fit_scene_landmarks.py` | Urban/rural annotated contact points |
| Fit a camera hypothesis | `scripts/fit_initial_camera.py` | Urban failure retained: fitting player worsens held-out NPC |
| Maintain sparse actor roots | `scripts/align_sparse_observations.py` | Two real later urban frames; explicit support-plane assumptions |
| Propose missing primitive | `scripts/propose_image_primitive.py` | Missing urban bollard; existing entities remain unchanged |
| Compile cached geometric evidence | `python -m whitebox.geometry_pipeline` | Urban cached VGGT/SAM inputs; compiler/render ran locally |
| Render candidate | `python -m whitebox.godot.render` | WSL Godot; use `--use-display` on this host |

Commands and real timing are in `local_tracking_and_depth.md`,
`local_sam_alignment.md`, `local_calibration_and_extension.md` and
`local_sparse_maintenance.md`. Numerical tools run with Windows Python/SciPy;
GPU tools run in the existing CUDA WSL Python. Rendering uses WSL Python/Godot.

Use camera/world coordinate projection before placing objects by world X. In a
camera facing positive Z, increasing world X appears left, not necessarily right.
The garage case demonstrates the cost of confusing world axes with screen axes.
Use ray/plane lifting only when support and camera assumptions are explicit.
Do not treat a cropped mask bottom as a foot/ground contact or an occluded facade
mask as the full building extent. Coarse actors intentionally need not copy
per-frame limb articulation.

Candidate success is not world acceptance. Existing `scene_refinement` tools
handle explicit evidence-backed geometry revisions, and `video_reconcile` handles
actual accepted video state/history. These local diagnostics do not replace
those commit paths or silently turn inferred geometry into observed facts.
