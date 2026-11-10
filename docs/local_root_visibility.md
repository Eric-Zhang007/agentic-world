# Contact visibility in sparse maintenance

2026-11-10. `align_observation` now requires an explicit `contact_visibility`
annotation before lifting an actor pixel onto its supplied support plane.

- `visible`: validate a finite in-bounds pixel, then update the coarse actor root
  and translate existing pose/held-weapon geometry with it.
- `occluded` or `unknown`: keep actor state unchanged and return a
  `skipped_actor_roots` entry. An absent field means `unknown`.

This is a tool-level check on supplied evidence, not an occlusion detector. It
does not prove the support plane or visibility annotation correct. A skipped
update means insufficient evidence, not that the actor remained stationary.
The helper still generates an uncommitted candidate; it does not accept video,
advance authoritative history, or synthesize a hidden motion path.

## Real image checks

The two urban later frames were visually reviewed again. Their annotated contact
regions are visible; `configs/urban_sparse_maintenance.json` now records this
explicitly. Re-running both frames produced
`outputs/local_visible_root_maintenance_20261110` (about 9.2 and 4.5 ms for the
numerical updates). Static reprojection residuals remain roughly 2-5 pixels.
No source images or accepted worlds were changed.

`configs/rural_occluded_root_review.json` tests the actual rural input image.
The player feet are visible; the NPC lower body is occluded by the player.
The NPC's mask-bottom pixel is deliberately supplied with `occluded`, so it is
shown as skipped, not used as a foot measurement. Output:
`outputs/local_occluded_root_review_20261110` (5.9 ms numerical update).
The output overlay was inspected and exact entity equality verified for the NPC.
This is a same-frame diagnostic, not a new temporal rollout. Its static anchors
were previously fitted on that image, so their near-zero residual is not an
independent camera-validation result.

`counterfactual.json` records where naive mask-bottom lifting would have placed
the NPC. It is diagnostic only and was never applied.

## Tests and callers

`scripts/test_root_visibility.py` covers skipped updates with a changed camera,
pose/weapon preservation, valid visible updates, invalid visibility and invalid
pixel coordinates. Existing synthetic callers explicitly label their generated
contact points visible. `scripts/align_sparse_observations.py` labels skipped
points separately and supports skipped rows without a pixel.

This change does not yet address partial-body height estimation in the separate
point-cloud asset compiler; those outputs remain inferred visible extents.
