# Geometry Category Contract

Implemented at the perception-to-compiler boundary in
`scripts/compile_geometry_assets.py`. No detector, renderer, database, accepted
world, or low-level geometry fitting module was changed for this fix.

## Root Cause

The installed Transformers GroundingDINO processor was inspected directly on
the server (Transformers 5.18.0,
`transformers/models/grounding_dino/processing_grounding_dino.py:143-225`).
`post_process_grounded_object_detection` takes sigmoid token probabilities,
selects tokens above `text_threshold`, and batch-decodes those tokens into
`text_labels`. Its `text_labels` candidate-list argument is explicitly unused.
The returned `labels` field contains the same strings, not category IDs. Box
confidence is the maximum token probability, not a disambiguated class score.

`scripts/probe_perception.py` saves these strings as observation `category`.
The cached observations retain neither candidate IDs nor token-to-candidate
spans/scores. A decoded phrase such as `road sidewalk` can therefore span
multiple prompt candidates. It cannot safely be assigned to one class, even
though both words are ground-related. The same problem affects `building wall`
and mixed ground/entity labels. This is not an urban-scene special case.

Previously the compiler passed all strings through. Ground classification used
exact equality, while all other classes fell through to a solid box. Facade
classification then used keyword membership. These inconsistent rules turned
uncertain labels into authoritative geometry roles.

## Contract

`map_geometry_category(observation)` returns a structured mapping:

```json
{
  "raw_label": "road sidewalk",
  "normalized_label": "road sidewalk",
  "canonical_category": null,
  "geometry_role": "unresolved",
  "accepted": false,
  "mapping_method": "exact_taxonomy_v1",
  "reason": "no_unique_supported_category"
}
```

Here `accepted` means accepted as a semantic input to this compiler, NOT world
acceptance. The original label, observation ID, source mask ID and mask path are
retained. Confidence is preserved but cannot override unresolved semantics.

Normalization only case-folds, trims whitespace/trailing periods, and collapses
whitespace. Exact category membership determines the role:

| Role | Exact categories |
|---|---|
| Ground surface | road, ground, sidewalk |
| Observed facade | building, wall |
| Actor | person, pedestrian |
| Entity proposal | car, truck, bus, motorcycle, vehicle, bicycle, tree, pole, dog, traffic light, bench |
| Excluded | sky, unknown |
| Unresolved | Everything else, including missing/non-string labels and unregistered composites |

The supported vocabulary covers the current geometry prompt and project
default perception categories. It is intentionally an explicit compiler
capability table, not an open-ended English classifier. New categories must be
registered deliberately with tests. Valid multiword classes such as
`traffic light` are not rejected merely for containing spaces. Substring,
keyword, first-token, synonym and all-ground-token heuristics are not used.
The facade branch now uses the resolved role, not split-word membership.

`partition_geometry_masks(masks)` separates accepted compile inputs from
non-solid semantic candidates and ground-completion blockers. IDs are not
renumbered after rejection. Unresolved/excluded masks:

- Never reach `compile_assets` as entity or ground inputs.
- Remain in `source_candidates.json` with `solid=false`, `accepted=false`,
  `UNRESOLVED_SEMANTICS` or `EXCLUDED_NON_GEOMETRIC`, and source identity.
- Retain processed confidence-filtered and raw silhouette masks in
  `semantic_candidate_masks.npz`, addressed by explicit keys in the candidate.
- Remain unknown/static blockers for ground completion; rejection does not
  turn their pixels into empty space available for inferred ground.

`category_mapping.json` records every observation, the taxonomy and policy.
The compiler report points to this file and counts accepted/unresolved/excluded
observations. Surviving entity geometry evidence includes its category mapping.
Ground evidence now lists source mask IDs/categories and visible pixel count.
Semantic evidence is saved before geometric fitting: if no supported ground
exists, compilation fails without a scene but still leaves the candidate list,
mapping and rejected-mask evidence. RGB correspondence validation still runs
before these artifacts are written.

A future detector integration should persist the prompt candidate inventory,
stable candidate IDs, token spans/query identity, per-candidate evidence and an
explicit unresolved decision before converting to display text. Merely passing
`text_labels` to this installed processor will not achieve that. This task does
not invent those missing fields for old caches or modify the detector API.

## Real Cached Verification

Recompiled the unchanged outputs from the fully fresh GPU2 run:
`outputs/geometry_proposals/fresh_gpu2_20261104_02/reconstruction.npz` and
`perception/result.json`, using the same RGB, 0.005 plane threshold and explicit
1.7m height prior. No new inference, manual road prompt, label edits, scene
patches, accepted-world writes or paid APIs.

First fixed result:
`/home/star/agentic_world/outputs/geometry_proposals/category_contract_20261104_01/`.
Elapsed 4.767s: compile 1.418s, render 3.222s, remainder RGB checks/overhead.
The same compilation was repeated after the Pi review-image return integration:
`/home/star/agentic_world/outputs/geometry_proposals/category_contract_20261104_02/`.
Elapsed 4.422s: compile 1.417s, render 2.921s.

| Check | Actual result |
|---|---|
| `geo_007`, road sidewalk | No entity; non-solid semantic candidate, 50,981 confident / 52,280 raw pixels retained |
| `geo_015`, building wall | No facade inference; non-solid candidate, 5,536 confident / 6,169 raw pixels retained |
| Entities | 80 before, 70 after; one thick box and nine ambiguous facade parts withheld |
| Remaining entities | IDs, positions, sizes, yaw, categories and solid flags unchanged for all 70 |
| Source camera | Unchanged |
| Ground mesh | `ground.bin` byte-identical before/after the category fix |
| Ground support | Only `geo_019` sidewalk; 2,584 visible pixels, 4,271 triangles |
| Ground inference | Zero inferred pixels/triangles; both persons and car rejected for no bracketed ground support |

The source-camera preview was visually inspected. The thick box no longer
occludes the lower bodies. **Road ground is still missing** and persons visibly
lack ground support; removing the false solid does not repair that missing
evidence. `building wall` is retained as an unresolved proposal, not interpreted
as a deleted building. This stricter rule is intentionally consistent across
ground and entity classes.

Latest main preview: `category_contract_20261104_02/preview/camera.png` relative
to the server geometry_proposals directory. Full four-view previews and
`candidate/category_mapping.json`, `candidate/source_candidates.json` and
`candidate/semantic_candidate_masks.npz` are available beside it.

## Tests and Pi Images

`scripts/test_geometry_category_mapping.py` has seven tests covering the whole
taxonomy, exact multiword membership, all ordered pairs of ground/actor/entity/
facade examples with four separator forms, unknown labels, immutable inputs,
mask/ID retention, a real compiler CLI success and a real no-ground failure.
The CLI tests verify that rejected pixels cannot become visible/inferred ground
and that semantic evidence survives a failed ground fit.

Local and server regression command: **53 tests pass**.

```bash
.venv/bin/python -B -m unittest \
  scripts.test_geometry_assets scripts.test_geometry_ground \
  scripts.test_geometry_regions scripts.test_geometry_category_mapping \
  scripts.test_geometry_pipeline
```

As separately requested, `propose` now returns exactly two absolute paths in
`review_images`, ordered `camera.png`, then `overview.png`; failures return an
empty list. Both views must be present in the renderer manifest and on disk.
Tests verify ordering/count, existence and rejection when the manifest omits a
required view. The latest real cached run returned these two paths successfully.
`world.ts` was not modified; its existing image-content channel can consume the
field directly.
