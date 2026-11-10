# Object visibility evidence

`memory/gta_visibility.py` separates geometry evidence from RGB visibility.
It does not change previous manual reviews, world state, or training labels.

Geometry categories are camera inside the oriented object bounding box,
near-plane intersection, entirely behind the near plane, outside the image,
projectable bounding box, and invalid geometry. A box that cannot be projected
is not automatically an invisible object. The near distance defaults to 0.05,
matching the existing pilot projection helper; this is a diagnostic convention,
not a verified match to the source renderer clipping plane.

Vehicle occupancy is recorded independently, including any carrier entity IDs
resolved from the scene's network IDs. Occupants are not automatically invisible:
windows, convertibles, and exterior seats can expose them. They require RGB
evidence before their appearance can be used as a visible reference.

Even a projectable box does not prove visibility. Static occlusion remains
unchecked because the required static collision resources are not available.
The module deliberately returns `rgb_visibility=UNKNOWN` and does not grant
appearance-reference approval. SAM/manual inspection remains a separate stage.

```powershell
python -m scripts.audit_gta_visibility --root E:/GTA_memory_prepared --output E:/GTA_memory_prepared/visibility_evidence_20261018
python -m unittest scripts.test_gta_visibility scripts.test_gta_record_joins scripts.test_gta_reference_candidates scripts.test_gta_object_memory
```

The batch audit reads previously exported observations, not every source frame.
Overlapping pilot packages are counted independently. Counts describe evidence
records, not distinct entities, clips, or unique training examples.
