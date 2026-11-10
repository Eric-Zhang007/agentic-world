# Local cross-scene review

2026-11-10. No remote server or paid model API. Original worlds unchanged.

## Reproduce

```powershell
python scripts/run_local_scene_review.py --cases configs/local_cross_scene_review.json --output outputs/new_cross_scene_review
```

This runs existing projection, coarse visibility and geometry checks. It consumes
perception bundles; it does not invoke perception models. Inputs and timings are
recorded per case. Output includes per-case JSON/overlays and `comparison.jpg`.
The actual reviewed output is `outputs/local_cross_scene_review_20261110`.

The rural bundle was freshly generated with local SAM2 in this run:
`outputs/local_rural_sam_20261110`. Existing local weights were used, with no
downloads or environment changes. Load time was 3.761 s, image encoding 0.479 s,
six box prompts 0.327 s. Peak torch allocated memory was 555.94 MiB, not total
GPU memory. Urban/garage perception bundles were reused from November 9.
Boxes, categories and world associations were supplied explicitly by Codex.

## Evidence and coverage

| Case | Reviewed world entities | Diagnostic time |
| --- | --- | --- |
| Urban | 4 of 26 | 0.088 s |
| Rural | 6 of 25 | 0.056 s |
| Garage initial frame | 4 of 21 | 0.063 s |

`unmapped_observation_ids` and `unreviewed_world_ids` expose coverage gaps.
Unreviewed does not mean missing, and zero unmapped observations does not mean
the detector found every visible object. These masks came from requested boxes.
All cases had no reported actor geometry collision, which does not establish
background correctness, visibility or temporal consistency.

## Visual findings

The rural source SAM overlay and all three diagnostic overlays were inspected.

- Urban parking-booth bounds have about 81 px center discrepancy. Its silhouette
  is cropped and includes attached architectural elements; the association to a
  single primitive is not a complete object-shape comparison.
- Rural player proxy top is too low in image space. NPC proxy is lower than the
  visible NPC region (57 px center discrepancy), but the NPC lower body is
  occluded by the player. Mask bottom is not a measured foot contact. This case
  must not be fixed by automatically moving the NPC to the mask bottom.
- Rural gun-shop mask includes sign/canopy while the mapped `gun_store` primitive
  has separate `store_sign` and `store_canopy` siblings. Its 48 px center error
  mixes component granularity and geometry. Group-aware comparison or explicit
  component masks are needed before a fit.
- Rural trunk masks touch the top image edge and include some crown pixels.
  They constrain visible placement, not full tree height.
- Garage initial NPC and bollard bounds are close (about 3 and 5 px center
  discrepancy). Player feet are cropped. The broad facade mask does not imply
  accurate room depth; the temporal garage experiment already disproved that.

The next general improvements should address observed-visible regions versus
complete proxy extents, explicit asset grouping, and support-point visibility.
Do not optimize all these differences away with actor heights or camera FOV.

## Files

- `configs/local_rural_sam_requests.json`: agent region prompts.
- `configs/local_cross_scene_review.json`: reproducible scene/bundle/mapping list.
- `scripts/run_local_scene_review.py`: batch review entry point.
- `scripts/diagnose_mask_alignment.py`: reusable diagnostic with coverage counts.
- `scripts/test_mask_diagnostic_bundle.py`: real temporary PNG bundle coverage,
  unknown/duplicate-ID rejection, and input-preservation tests.

## Group-aware comparison

The same diagnostic now accepts either one entity ID or an explicit list of
component IDs per observation. Example:

```json
{"gun_shop":["gun_store","store_canopy","store_sign"]}
```

Group projection uses the union of all member projected bounds before clipping.
If any member cannot be projected, the metric is unavailable rather than silently
dropping that member. Visibility rays use the nearest group member and count only
nonmember boxes as external blockers. Actual holes and disconnected parts remain
holes in the ray/box test; the bounding-box IoU itself is still not silhouette IoU.

Run `scripts/run_local_scene_review.py` with
`configs/local_group_scene_review.json`. The actual output is
`outputs/local_group_scene_review_20261110`, visually reviewed. No scene geometry,
camera, mask, or observation changed between the original and grouped runs.

| Observed object | Single-part center error / bbox IoU | Group center error / bbox IoU |
| --- | --- | --- |
| Urban foreground lamp | 13.55 px / 0.617 | 5.11 px / 0.793 |
| Urban parking booth | 80.96 px / 0.488 | 69.98 px / 0.581 |
| Rural gun shop | 48.04 px / 0.400 | 20.44 px / 0.504 |

The lamp base, booth fascia and shop canopy no longer count as external blockers
of their own assembly. The shop still has an external utility-pole blocker.
Reviewed member counts become 6/26 urban and 8/25 rural; garage remains 4/21.
These are not increased independent observations or proof that each component
is correct. The improvement is a correction to comparison granularity, not a
reconstruction gain. Remaining discrepancies are not repaired by this change.

Group mappings are explicit review inputs, not automatically inferred persistent
asset hierarchies. No fitting step consumes these metrics as an acceptance rule.
