# Full local GTA data release processing

Source snapshot: `E:/GTA_555`, 1134 segment manifests. All manifests enter the
source ledger. Missing camera/world/entity dependencies exclude 474 segments;
the other 660 segments enter complete media decode and actual record checks.
Never treat presence of a manifest or a candidate ID as a positive label.

Working release directory: `D:/GTA_memory_release_20261018` (SSD).
The earlier E-drive working snapshot is retained; final archives are copied to
`E:/GTA_memory_delivery_20261019` after validation.
Existing pilot packages and raw source remain unchanged.

The source index stores compressed Parquet row-group/row pointers in SQLite
and rejects ambiguous joins. Camera/world payloads are read directly from the
source files; initial payload-copy indexes are superseded working artifacts,
not part of the delivery. Source sizes must be stable during scans.
Every target/reference frame must have actual camera,
world and lifecycle support, valid camera rays, monotonic clocks and contiguous
target decoder indices. Camera-inside-object risk routes the whole candidate
out of the unchanged-proxy training set. Occupants without independent RGB
evidence are not used as appearance references.

One nonoverlapping 120-frame target grid is used with a historical frame 120
frames before each target start. Reference age comes from source timestamps,
not the nominal frame rate. Known subject clothing/component changes break
reference compatibility. Object IDs and model identities must remain supported
throughout each window. State is preserved per timestamp, not flattened into a
single permanent object appearance.

The configured visual model reviews each eligible window using the historical
reference and eight distributed target frames, with paired whitebox imagery
and explicit projected entity labels. It may reject a pair or individual
reference objects. Its judgments are estimated labels, not human ground truth.
At most four largest projected candidate objects per window are reviewed.
Objects must have a projected width and height of at least 24 pixels throughout
the selected frames. This biases the release toward large continuously
projectable objects; it is not an exhaustive scene annotation.
Coarse articulation and wireframe vehicles are acceptable; camera mismatch,
false foreground obstruction and identity confusion are not. Source media are
fully decoded even though visual model review uses sampled frames.

Accepted RGB reference boxes are segmented with cached SAM2.1 and encoded with
cached DINOv2 on the existing server, using a persistent independent worker.
Empty/low-quality masks do not become references. No models are trained and
no H3 weights/environment are changed. SAM masks remain model estimates.

The packaged loader separates conditions from target RGB supervision. HUD
regions are ignored by loss masks and blanked in reference RGB on loading;
there is no synthetic inpainting evidence. It is mandatory to apply loss masks.
Per-object memory edges record observed and used timestamps, exterior revision,
target interval, matching projection trajectory, and raw source state records.

Finalization decodes all exported clips again, validates every edge and asset,
checks every source segment/candidate has a terminal status, tests actual
dataset loading, and isolates entire episodes across train/validation/test.
Shared map scenery across independent episodes is not a novel-scene benchmark.

Building and ground instance-memory edges cannot be certified without stable
identity and static geometry. They remain scene context, not fabricated object
memory labels. This exclusion must appear in the acceptance report. Likewise,
120 frames are a model-independent training format; an H3-specific trainer may
need a documented temporal adapter. Do not claim a training run occurred.

The release is only complete when `DELIVERY.json` exists and all archives pass
integrity verification. Intermediate SQLite/index/review files are not the
delivery and must never be described as a finished training dataset.
