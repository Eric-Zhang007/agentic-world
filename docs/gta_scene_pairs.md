# Scene reference pairs at trajectory intersections

This dataset contains scene-centric frame pairs, not object-memory annotations.
Reviewed samples also contain 120-frame target RGB/whitebox videos. Only samples
with `training_status=REVIEWED_SCENE_VIDEO` enter the training split lists.
`samples.jsonl` lists all pairs, including diagnostic exclusions; do not use it
as an unfiltered training list. `training_review.json` records every decision.
Five target frames (0,30,60,90,119) were visually inspected by Codex for each
exported video, after full decode and first-frame pixel alignment checks.
Each sample contains two observation times at a nearby camera position and
similar viewing direction. At each time, original-resolution RGB and whitebox
frames are paired by the source alignment records.

## Files and training use

- `reference_rgb.png` and `reference_whitebox.png`: historical condition pair.
- `target_whitebox.png`: target geometry/camera condition.
- `target_rgb.png`: supervision only, never a reference condition.
- `prompt.txt`: deterministic scene-reference instruction, no target RGB caption.
- `reference_state.json` and `target_state.json`: original frame alignment,
  camera interpolation records, estimated source time, source paths/indices.
- `comparison.jpg`: reference RGB/whitebox on top; target RGB/whitebox below.
- `*_background_mask.png`: coarse dynamic-object and HUD exclusions used for
  background matching. These are not true segmentation masks or loss masks.
- `sample.json`: provenance, times, view distance/angle, feature evidence.

Paths inside each sample are relative to its folder. Absolute source paths in
state records are provenance and are not needed to load the supplied images.
`samples.jsonl` lists sample directories. Episode-isolated splits retain the
existing release partition. An empty split remains empty; do not mix adjacent
views across splits merely to fill it. No trainer-specific dataloader is assumed.

## Selection and evidence

Default candidate search uses camera distance <=5 world units, height difference
<=2, viewing direction difference <=35 degrees, and reference age >=30 seconds.
Frames are sampled every 120 source frames (nominally five seconds).

Same-camera leave/return requires a recorded excursion of >=15 world units and
no gap greater than 7.5 seconds in the sampled trajectory. Different cameras are
explicitly labeled as cross-camera intersections, not same-camera revisits.
Different episodes or server clocks are never joined. Stationary observations
are not labeled as leave/return. Nearby candidates are temporally/spatially
deduplicated. The thresholds are recorded in `mining_config.json`.

SIFT features are computed outside projected dynamic-object boxes and HUD
margins. Ratio-filtered unique matches are checked with a RANSAC homography.
Export requires >=20 inliers, >=0.35 inlier fraction, coverage in >=3 of 12 image
cells, and >=15% available background area in each image. This is inexpensive
automatic evidence of shared background, not a guarantee of perfect alignment,
unchanged scene state, or exhaustive quality. Repetitive textures can produce
false matches; substantial parallax can cause false rejection.

Input frames reuse the existing sequential exact-decoder-index PNG cache.
Source camera/world joins and frame exclusions are checked again. We do not use
random seeking, make paid API calls, infer object identities, or claim that every
sample has received human visual review. The previous object-centric dataset is
unchanged. Full original-resolution images are retained, including original HUD.

## Commands

Run in the original agentic_world project (NumPy, SciPy, OpenCV, orjson,
PyArrow and existing project helpers are required for extraction):

```powershell
python -m unittest scripts.test_gta_scene_pairs
python -u -m scripts.build_gta_scene_pairs
python -u -m scripts.package_gta_scene_pairs
```

Only an image reader and JSON parser are required to consume the delivered data.
Included processing code documents the added implementation; extraction also
depends on the original project's camera/record helpers and original GTA data.
