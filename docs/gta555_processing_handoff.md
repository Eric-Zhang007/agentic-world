# Full source processing operations

Working root: `E:/GTA_555_complete_rebuild_20261025`.
The previously delivered archive is unchanged. The new full dataset is not yet accepted.

## Active stages

- Source recovery: `recovery/RECOVERY_STATUS.json`, `recovery_active_error.log`.
- Byte-index adaptation: `record_adapter/INDEX_STATUS.json`, `record_adapter_active_error.log`.
- Verified proxy synchronization: `proxy_cache/SYNC_STATUS.json`, `proxy_sync_resumed_error.log`.
- RGB/camera qualification: `passages/PREPARATION_STATUS.json`, `passage_preparation_active_error.log`.
- Automatic survey, background matching and candidate export: `passages/TRAINING_PIPELINE_STATUS.json`.

The continuation process waits for all source preparation to finish. Candidate export
does not confer visual acceptance. Its final pending-review state is intentional and
must be followed by actual review and balanced mixture construction.

## Verified repairs on October 26

Proxy receipts contained `original_proxies/.../proxy.mp4`, relative to the server
render working directory. SFTP interpreted these from the login directory and
retried a nonexistent path forever after 1,134 locally reusable proxies. The sync
now resolves each path from the explicit absolute remote cache root and its
`output_relative`. Actual downloads advanced beyond that point after restart.

The packager now takes `--mixture` and `--output`, derives counts from manifests,
and generates documentation without the previous release's fixed counts. A real
four-task subset of previously accepted media passed ZIP integrity and portable
loader checks in `D:/GTA_source_work_cache/package_window_check/archive_check`.
This is a packaging exercise, not the new dataset release.

## Expanded visual exercise

`review_expanded` freezes 120 qualified source segments and 1,420 trajectory points.
The matcher tests approximate revisits including cross-camera passages. Three
real 120-frame training examples were exported. Inspection of historical pairs
and target frames 0,30,60,90,119 accepted two outdoor examples and rejected one
indoor example whose whitebox omitted counters, shelves and enclosing surfaces.
Evidence and decisions: `review_expanded/pending_audit_camera_passages`.

The completed subset search yielded 298 candidate pairs: 138 cross-camera,
35 same-camera returns and 125 nearby-history pairs. Only two references are
older than 120 seconds; these candidates are not all long-term revisits.
The full 298-candidate video export runs separately with logs at
`review_expanded/export.log` and `export_error.log`. It resumes existing results
and preserves the three explicit visual decisions above.

The rejected exact target window is also recorded under
`passages/visual_review/ep_20260916_123157_7d08_agent_001_segment_00004_w010/decision.json`,
so the full exporter excludes it. This decision is not generalized to uninspected
windows. Coarse vegetation and approximate actor poses remain known limitations.

The packaging exercise also executes the unpacked `example.py` against real
media. It uses the always-present train manifest rather than requiring an
optional balanced-round manifest. The latest verified exercise is
`D:/GTA_source_work_cache/package_example_check/archive_check`.

## Remaining delivery work

Finish recovery and qualification, mine and export the complete candidate pool,
visually review reference/target correspondence, construct episode-isolated
scene/object mixtures, and package the accepted pool. The old finalizer still
has old input defaults and produces separate intermediate archives; it must not
be invoked unchanged as the new unified delivery command. Use the parameterized
packager only after the new mixture has been validated.

```powershell
C:\Python314\python.exe -m scripts.package_gta_unified --mixture <validated-mixture-directory> --output <new-delivery-directory>
```

No source file count, process existence, or successful proxy decode alone is
evidence that RGB/whitebox visual correspondence passed.

## Late-evening check, October 26

All 1,992 verified proxies are now available locally. The 298-candidate exercise
finished export: 291 pending visual review, two sampled-review acceptances,
one visual rejection, and four camera-invalid exclusions. Remaining review
sheets are under `review_expanded/pending_audit_remaining`.

An older episode used `sample_seq_estimate` instead of `client_sample_seq`.
The original package's `tools/verify.py` explicitly supports that fallback.
The adapter now follows the same rule while retaining byte hashes and index
identity checks. Four adapter tests passed, and the actual affected episode
`ep_20260909_132205_d2da` then adapted successfully (world 3,605 records;
agent_000 3,370; agent_001 3,579). Index progress reached 13 of 21 episodes,
with the remaining eight waiting for their spool inputs.

## October 27 recovery and available-source processing

The overnight recovery ended with 324 OSS EOF failures, leaving approximately
192 GB missing. Direct OSS access was tested successfully. Recovery resumed
with HTTP/HTTPS/ALL proxy variables cleared and NO_PROXY set for direct access.
New RGB files actually completed, rather than merely reporting live processes.
Logs: `recovery_direct.log` and `recovery_direct_error.log`.

`resume_gta555_recovery.py` waits for the active pass to finish and retries a
failed pass up to three times, with a delay and no concurrent recovery writers.
It stops on conflicts or exhausted retries. It does not detect a hung process
whose status remains recovering. Logs: `recovery_retry.log` and its error log.

`available_source_training` is a frozen subset pipeline independent of missing
inputs: 1,150 qualified segments, 13,429 points from 1,282 processed segments.
Its continuation runs survey, matching, video export and review sheet creation.
It uses qualification records from `passages`. The full-source continuation is
still waiting for all inputs; do not combine overlapping subsets by concatenation.

Actual visual decisions now cover indices 0 through 63 of
`review_expanded/pending_audit_remaining/index.json`: 55 accepted and nine
excluded. The other 227 remain pending. Review JSON files name inspected sheets,
specific rejected indices, reasons and limits. The review tool now supports
explicit partial index lists, preserves previous ledger entries, and leaves
unseen samples untouched. Seven focused regression tests passed.

Partial acceptance is still reference-pair plus five-frame inspection, not
every-frame video verification. Close solid-proxy occlusions, yard structures
replaced by walls, and inconsistent roadside silhouettes are rejected. Coarse
limb poses, vegetation and missing distant terrain remain limitations, even in
otherwise accepted near-field scene references.

## October 27 evening: reboot recovery, completed subset review, index repair

Windows rebooted at 19:45:07. All previously listed processing PIDs had exited;
their status files were stale. Recovery, index adaptation, available-subset
mining and full-source continuation were restarted. Recovery advanced by over
26 GB across 69 files after restart, with zero new failures at that check.
Index adaptation reached 14/21 episodes. Verify live counters again before
reporting progress; these numbers are snapshots.

All 298 candidates in `review_expanded` now have a decision: 247 passed sampled
paired-media QA, 47 failed visual QA, four failed camera validation. Accepted
types: 118 cross-camera passages, 32 same-camera returns, 97 nearby histories.
Only ONE accepted reference is at least 120 seconds old. 244 accepted examples
come from ep_20260916_123157_7d08 and three from ep_20260917_125015_65e7. This
small, concentrated exercise is NOT the final balanced dataset and must not be
advertised as broad long-term coverage. Inspection covered historical RGB/proxy
pairs and target frames 0/30/60/90/119, not exhaustive temporal inspection.

`reuse_gta_passage_reviews.py` carries decisions forward only when reference and
target IDs, timestamps, cadence and all six paired image/video files match
exactly. Changed references stay pending. Both continuations now invoke this
before creating review sheets. Existing snapshot data and review indexes survive
restarts. Source extraction and the prior delivered ZIP remain unchanged.

The reboot left a corrupt NPZ feature cache. Matching now closes failed loads,
recomputes corrupt entries and publishes cache files only after writing completes.
The resumed matcher passed its former failure and reached 5,700 targets with
1,057 automatic candidates at the latest check (not visually approved samples).

A separate real data bug was found: October segmented MKVs retain episode-global
`rgb_decode_index` alongside file-local `rgb_segment_frame_index`. Using the global
index on a segment falsely excluded valid footage. `rgb_file_index` now selects
the explicit local field, falling back to the old field for earlier inputs. Raw
source state is preserved; trajectory points also retain `source_rgb_decode_index`.
Preparation, cached frames and passage export use the same local indexing rule.

There were 136 MEDIA_COUNT_MISMATCH results. `recheck_gta_segment_indices.py`
rechecks only exclusions with differing explicit local/global indices, archives
their old summaries under `before_local_index_fix`, then resumes qualification
with `--resume-watch`. Do not run a second qualifier during this repair pass.
Logs: `local_index_repair.log` and `local_index_repair_error.log`.

The first actual repaired segment became QUALIFIED with 12 points. Its first
120-frame export passed decode/endpoint checks but showed a full-screen map UI;
that exact target was visually rejected in `passages/visual_review`. See
`local_index_export_check`, which is an integration artifact, not a training set.
Thirteen focused tests passed, including segmented-index precedence and preserving
global provenance, partial review, exact-media reuse, index adaptation and recovery.

Live logs after this reboot: `recovery_after_reboot*.log`, `index_after_reboot*.log`,
`available_cache_closed*.log`, `available_source_training/match.log`,
`local_index_repair*.log`, `full_after_reboot*.log`. Full preparation status is
stale while the repair pass owns qualification; use repair log results until it
hands back to the watcher. Final mixture and unified packaging remain unfinished.
