# Garage RGB candidate revision

Final candidate: `outputs/local_garage_revised_20261110/scene_02.json`.
Review image: `outputs/local_garage_revised_20261110/render_02/camera.png`.
Intermediate scene and actual render are retained as `scene_01.json` and
`render_01/`. Original scene and source files were not edited. No GT, alternate
scenes, network, model API, server, or subagents were used. No core changes.

## Evidence and changes

Read `docs/local_whitebox_tools.md`; visually inspected the specified case_06
RGB and both new Godot camera renders. Used fresh SAM bundle 01 with the existing
explicit garage mapping and `scripts/diagnose_mask_alignment.py` before and
after. Reports and diagnostic overlays are in `before/`, `after/`, `after_02/`.

Corrected inferred layout convention: +X is screen-left for this +Z-facing
camera. Mirrored entity X coordinates, then fit camera perspective against
approximate entrance endpoints. Lower opening targets are (487,484) and
(1135,638). Upper targets are offscreen extrapolations, NOT observed corners;
their uncertainty limits this calibration. `anchors.json` records the targets
and projections. Lower-edge errors after fitting are about 38 and 6 pixels.
The before-opening projection in that file uses the revised coordinate points
with the old camera, so it is a camera diagnostic, not a full old-scene score.

Kept GTA coarse actors without joint reconstruction. Actor roots are ground-plane
hypotheses; height is allowed to change in revision 02. NPC shoe contact is an
approximate visible annotation. Player root target (520,790) is an explicitly
unobserved offscreen prior. The image bottom was never treated as foot contact.
Rear object centers were repositioned using rays at assumed Z depths, not metric
depth recovery. Expanded left facade and retaining wall to cover cropped extents.
Hidden thickness, room depth, roof, side walls and rear wall remain authored
completions. Their physical correctness is not established by this image.

## Before/After

These are visible-mask bounding boxes versus unoccluded projected coarse bounds,
not silhouette scores, visibility checks, or evidence of correct body pose.

| Entity | Center error before / final px | Bbox IoU before / final |
| --- | --- | --- |
| Player | 238.81 / 19.91 | 0 / 0.777 |
| NPC | 173.95 / 2.88 | 0 / 0.813 |
| Left facade | 897.26 / 29.89 | 0 / 0.840 |

## Visual Judgment

Substantial real improvement in handedness, screen placement and actor scale,
but STILL FAILING as a complete garage reconstruction. The opening now occupies
the correct right-hand region; player head is near the observed height, with
legs leaving frame; NPC occupies the correct neighborhood. Rear door/cabinet
and colored containers have improved screen ordering.

Important failures remain: the inferred left interior wall hides the repositioned
workbench; expanding the facade obscures much of the bollard. Thus higher bbox
scores conceal visibility failures. The facade roofline exposes sky where the
source continues as building. Left entrance base remains too high. Rear door
scale and room depth are uncertain. The player is rigid and narrower than the
walking silhouette; arms and clothing are not reconstructed. Grid floor,
lighting, ceiling detail and wall appearance differ strongly from RGB. These
are not merely texture errors: side-wall extent and occlusion require revision.
No authoritative world acceptance is claimed.

## Timing and Reproduction

Total hands-on work approximately 4 minutes, including reading, annotation,
numerical fitting, rendering, inspection and this report. This is an estimate:
no start stopwatch was captured. Script creation was 00:06:48 local time and
final inspection was around 00:08:09 on November 10, 2026.

Exactly TWO WSL Godot render invocations, both `--use-display`, 1280x720,
four review images per invocation. Measured subprocess wall times: 8.263 s and
5.997 s (14.260 s total). Both exited 0; llvmpipe renderer. Timing JSON files
are in the output directory. No third render was attempted.

`revise.py` and `refine.py` retain the numerical construction and commands.
They are historical run scripts, not idempotent jobs: diagnostic output folders
must be fresh on reproduction. Final scene is `scene_02.json`; do not substitute
the intermediate scene based only on filenames. No tests of the concurrently
edited fitting module were run or changed.
