# Raw-spool background-memory increment

This increment recovers 474 paired segments from four recording batches that
were previously skipped because derived Parquet records were absent. Raw
client/world JSONL offsets are indexed without changing the source files.
Camera interpolation and frame alignment use the recorded clock domains and
sample/tick IDs. Unaligned frames are excluded from eligible target windows.
The separate 1007 batch has no locally available paired whitebox videos and is
not included merely because raw camera/world records exist.

## Processing

1. `prepare_gta_spool_passages` decodes both videos fully, checks frame counts,
   resolves camera/world records, and caches exact five-second samples and
   target-window endpoints. The source report counts decoded and invalid frames.
2. `survey_gta_passage_pairs` searches strictly historical camera-nearby
   references in the same episode and server clock domain. `match_gta_passage_pairs`
   tests background SIFT evidence after masking projected dynamic objects and
   HUD margins. This is an automatic overlap test, not instance annotation.
3. `export_gta_passage_training` checks all 120 target frame alignments, exports
   RGB/whitebox clips, fully decodes each clip, and compares exact first/last
   frames to cached source frames. The minimum endpoint PSNR is 26 dB.
4. `review_gta_passage_targets` places the historical RGB/whitebox pair beside
   target frames 0, 30, 60, 90 and 119 and records explicit visual decisions.
   Pending and rejected samples cannot enter the training manifests. The batch
   layout and per-sample review metadata record the exact visual-review scope.
5. The finalizer preserves previous archives, merges eligible manifest rows,
   exercises the real mixed loader, and reads back both new ZIP archives.

Run from the project root:

```powershell
python -m scripts.prepare_gta_spool_passages --workers 8
python -m scripts.run_gta_spool_cleaning
# Review each pending_audit batch before finalizing.
python -m scripts.finalize_gta_passage_training --root D:/GTA_spool_passages_20261020 --delivery E:/GTA_spool_passage_training_20261020 --episode-splits configs/gta_spool_episode_splits.json --previous-scenes D:/GTA_passage_pairs_20261020/training
```

## Contract and limitations

The condition is an earlier RGB/whitebox frame pair, a later whitebox video,
and a fixed background-reference prompt. Supervision is the aligned later RGB
video. These are scene-background samples; they do not claim object identity,
static-object visibility, appearance-state transitions, or dense geometry.
Scene JSON placement membership is not a visibility label. Existing whitebox
geometry is reused, not reconstructed or made more detailed by this cleaning.

Keep `nearby_history`, `same_camera_return`, and `cross_camera_passage` separate.
The first is continuous nearby context, not proof of leaving and revisiting a
location. Existing object-memory samples are retained. Balanced rounds start
at 50% background, 20% person, 20% vehicle and 10% mixed objects, with nearby
history at most half of the background portion. Ratios are not training-tested.
Splits isolate recording episodes, not GTA map locations. Reference matching
uses automatic features; target temporal visual review is sampled, not frame-by-frame.

Authoritative counts are in distribution_report.json, DELIVERY.json, and the
per-batch review ledgers. No paid inference APIs or model training are used.
