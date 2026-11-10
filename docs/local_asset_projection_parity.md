# Asset projection parity

2026-11-10. Found two diagnostic mismatches against the existing Godot renderer:

1. Actor height projection ignored the persistent posture weights even though
   Godot scales the catalog asset by the corresponding height factor.
2. The actor's nominal size was used as its rendered box. The catalog human has
   arms extending beyond its nominal 0.65-unit width, so that box did not contain
   the rendered asset.

`whitebox/asset_geometry.py` now exposes local vertices with the same catalog
scaling and actor/box origins as `render.gd`. Projected bounds use those vertices.
Coarse visibility uses their local oriented envelope, still not exact mesh-ray
intersection. Noncatalog boxes retain their previous bounds. Actor landmark
fractions also use the existing posture-adjusted size.

The actor collision/movement envelope was intentionally not changed: collision
policy and visual bounds are different uses. No assets, standing scene geometry,
accepted worlds, or history were modified. The standing/coarse-posture renderer
itself is unchanged; this does not introduce full prone animation or fine joints.

## Actual local render check

`outputs/local_posture_projection_20261110` contains isolated colored catalog
actors for standing, crouched and low_guard. All three were actually rendered
with local Godot/llvmpipe. Red mesh pixels are thresholded only in this controlled
synthetic scene, not used as a detector for real video. The rendered frames were
then reused to check the corrected diagnostic without rerendering unchanged
geometry. `catalog_comparison.jpg` was visually inspected.

| Posture | Bounds IoU against rendered color mask | Maximum edge discrepancy |
| --- | --- | --- |
| Standing | 0.9932 | 0.240 px |
| Crouched | 0.9891 | 0.426 px |
| Low guard | 0.9913 | 0.240 px |

This checks rendered asset versus diagnostic math, not GTA reconstruction quality.
Earlier unscaled boxes had about 26/43 px vertical-center discrepancies for the
crouched/low_guard cases. The corrected actor bounds also cover the actual arms.

Three real source cases were rerun with saved SAM bundles under
`outputs/local_catalog_scene_review_20261110`. No new perception was claimed for
that rerun. Old diagnostic reports are retained, but their actor bbox numbers
should not be compared as though all used the corrected catalog bounds.

Tests in `scripts/test_posture_projection.py` cover height factors, blended
postures, fixed root, catalog vertex projection, ray-envelope height and unchanged
nonactor boxes. Existing visibility, camera and matching diagnostics remain in
the regression run.
