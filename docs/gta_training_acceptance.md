# Training acceptance and current limitations

The October 29 audited manifests contain 7,085 accepted task records. Manifest,
dependency and representative loader checks passed. Archive and upload completion
must be read from `unified_delivery/DELIVERY.json` and `HF_UPLOAD_STATUS.json`;
manifest completion alone is not packaged delivery completion.
Sampled visual approval means that the historical pair and five target pairs
have acceptable coarse correspondence. It does not establish complete temporal
alignment, reference usefulness throughout the target, or balanced coverage.

## Intended supervision

Input: earlier RGB/whitebox reference pair, target whitebox sequence, and a
description consistent with the recorded scene. Target: aligned RGB sequence.
Scene samples do not claim object identity labels. Recorded target outcomes and
auxiliary state-change annotations are for auditing/sampling, not prompt inputs.

## Required release evidence

- Account for each segment in the local package's data/segments.json. Distinguish
  absent local files, missing adapted records, preparation, and candidate export.
  Cloud inventory is not the denominator for this local task.
- Verify timestamps, camera conventions, local video frame indices, readable
  paired media, and reference chronology within the same clock domain.
- Check useful shared static background between reference and target. Nearby
  cameras and SIFT inliers are candidate evidence, not sufficient acceptance.
- Review the actual historical reference pair and target together; do not inherit
  acceptance from a different reference. Reject UI contamination and major
  geometry/occlusion disagreements. Preserve the scope of sampled inspection.
- Deduplicate merged old/new candidates and prevent overlapping source windows
  from crossing splits. Report episode isolation separately from unseen-location
  evaluation; they are not equivalent.
- Produce realized counts and sampling proportions by scene, object task,
  episode/location, time gap and view angle. Keep nearby continuation separate
  from leave-and-return memory. Rare labels must not imply broad coverage.
- Load the packaged samples with the delivered loader and check every referenced
  media path. Keep the old release unchanged until the new package passes.

Training effectiveness is not yet measured. Passing data acceptance establishes
usable supervision, not a guarantee that a trained renderer will use memory.

## October 29 corrections

The finalizer now rejects inherited target-only review and requires explicit
historical-pair review. This closes one eligibility error; it does not complete
the other release checks. The local coverage audit reads the published segment
list directly and does not consult OSS.

The local audit found 1,893 published RGB paths present out of 1,992 declared
segments. All segment frame/manifest/validation files and proxy receipts were
present. Adapted record receipts covered 1,373 segments; 1,363 segments were
prepared. These are presence/preparation counts, not accepted training counts.

The earlier candidate snapshot covers 1,150 qualified segments. Incremental
processing in `additional_local_training` snapshots currently qualified segments
and excludes earlier target point IDs during candidate search. Earlier points
remain eligible historical references. This preserves prior exported samples and
reviews. Both source pools now have a recorded decision for every eligible scene.
The accepted scene pool contains 2,101 records; 1,407 direct local visual approvals
and 694 retained prior API approvals. Earlier object QA is retained separately.

Final merging now removes duplicate scene sample IDs across old/current sources
and rejects episode split conflicts. This does not detect different IDs with
overlapping source windows; that remains a separate release check.
