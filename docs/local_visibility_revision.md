# Garage candidate revision with visibility evidence

## Actual construction progress

`outputs/local_garage_visibility_fix_20261110/comparison.jpg` compares source,
initial candidate, projection-only revision, and visibility-corrected revision.
All proxy images are actual Godot renders, not illustrations.

An astra-low subagent revised the initial source-only scene in a separate output
directory. It corrected screen/world handedness, camera framing and actor
placement. Player/NPC center errors dropped from 239/174 pixels to 20/3 pixels.
But visual inspection still failed: NPC and workbench were obscured by an
incorrectly placed interior wall, despite good projected bounding-box overlap.
The main thread additionally ran `validate_geometry`, which reported NPC/left_wall
collision. These facts were not visible in bbox metrics alone.

## Reusable visibility check

`whitebox/visibility.py` casts camera rays through observed mask pixels and tests
Y-rotated proxy boxes. For pixels also intersecting the target's coarse volume,
it reports which static box is in front. `diagnose_mask_alignment.py` now includes
this result and current actor/ground geometry errors automatically.

The revised garage had 58/279 comparable NPC pixels blocked by `left_wall`.
The bollard was misplaced, and all 14 pixels overlapping its old proxy were
blocked by `low_wall`. The tiny overlap is recorded; 100% here does not mean
every source bollard pixel had a corresponding proxy hit.

This is not exact mesh visibility. Articulated limbs, terrain surfaces and
non-box blockers are not modeled by this diagnostic. A foreground structural
trim may also legitimately block a larger wall's coarse volume. Blocker reports
must be interpreted with source masks and actual renders, not auto-deleted.

## Evidence-backed candidate changes

`configs/garage_visibility_refinement.json` records the specific evidence and
uncertainty for edits. The existing `scene_refinement.apply_proposal` applied it
to the uncommitted candidate. It widens the inferred room behind its entrance,
keeps roof/back wall consistent, puts below-floor furniture on the floor, and
places the bollard through its source ground-contact ray. Hidden room extent
is an authored hypothesis, not a recovered measurement.

Final candidate: `outputs/local_garage_visibility_fix_20261110/scene.json`.
Actual four-view render completed in 8.43 seconds including WSL command overhead.
Camera image was inspected: NPC, workbench and bollard are now visible.
NPC blocked count is 0/279. Geometry errors are empty. All actor values and the
camera are exactly unchanged from the projection-only revision, so the change
does not obtain visibility by moving the observed actors elsewhere.

The scene is still not a complete reconstruction. The facade roofline exposes
sky absent from the source; rear furniture depth/extent and room dimensions
remain uncertain. Actor posture/clothing are coarse, intentionally not fitted
joint by joint. No accepted world or history was changed.

## Fitting bug fixed

The landmark fitter previously allowed a linked part to simultaneously receive
an independent optimization parameter. The later assignment silently overwrote
the group translation. A failing regression reproduced this before the fix.
The fitter now checks the complete set of written entity/field/axis targets and
rejects overlaps before optimization. Shared translation of disjoint parts
continues to pass its original synthetic test.
