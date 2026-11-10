# Local geometry diagnostic

Date: 2026-11-09. Workspace: C:/Users/zjc/omgd/agentic_world.

## Scope and existing state

Read geometry_assets.py, geometry_ground.py, geometry_regions.py and existing
geometry tests before editing. This directory has no Git repository metadata:
git status failed, so pre-existing edits cannot be identified by a baseline diff.
Preserved existing code and tests; no subagents, server, or paid API were used.

## Reproduced bug: observation IDs changed occlusion geometry

complete_ground excluded other dynamic silhouettes by comparing string IDs.
Two distinct observations with the same ID therefore failed to block each other.
IDs converted to strings can also collide between numeric and textual inputs.
This is independent of scene category layout or metric scale.

Regression constructs a perspective ground plane with adjacent person and car
silhouettes. Changing only the car ID from `actor` to `different` changed 340 of
4800 inferred-mask pixels before the fix. The duplicate-ID case filled those
pixels, while the distinct-ID case correctly rejected unbracketed support.
The new regression failed before the implementation edit (1 test, 0.138 s;
command wall time 0.880 s).

Fix: identify the current dynamic observation by its list index when excluding
other silhouettes. External IDs remain unchanged in evidence records.

Validation command:

```text
python -m unittest scripts.test_geometry_ground scripts.test_geometry_assets scripts.test_geometry_regions
```

Result: 36 tests passed in 0.679 s (command wall time 1.120 s). Includes the new
ID-invariance regression and existing asset, facade, and ground tests.
geometry_assets.py and geometry_regions.py were inspected and tested but did not
need edits for this fix. No scene-specific thresholds or heuristics were added.

## Rural cache availability

All 7 JSON files parsed and all 12 PNG/JPG files passed Pillow verify; inspection
command wall time was 0.259 s. Candidate and refined scene/report files and
previews can be reused locally. The perception JSON contains 20 observations;
mask paths point to /home/star/agentic_world/outputs/geometry_proposals/
rural_same_image_20261104/perception/. No mask images or reconstruction NPZ
exist under outputs/rural_fast_geometry (recursive inventory).

Consequently this cache alone cannot rerun point-map geometry fitting or ground
completion. Existing preview images do not validate current rendered geometry.
The MP4 was inventoried but not decoded. This task did not modify cache files,
run a renderer, or measure rural reconstruction accuracy; rendering belongs to
the main thread. Synthetic regression evidence establishes the algorithmic bug,
not a claim that this particular rural cache contains duplicate IDs.
