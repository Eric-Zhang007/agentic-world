# Follow-Up Validation

Work performed on server `/home/star/agentic_world`, 2026-10-03 Asia/Shanghai.

## Data Availability

Read-only inspection of six GTA archives in the supplied WeChat file directory
(GTA_recorder2, GTA (3), GTA (4)(1), GTA (4), GTA-3 (2), GTA_0906 (2))
found no world_state JSONL/Parquet, session_metadata.json, static.sqlite3 or
recorded rgb.mp4. The bounded server search also found no paired payload.
Obtaining the original episode directory is still required for measured 3D
projection calibration. No recorded world facts were fabricated.

The inverse package does contain original sampled RGB JPEGs with timestamps.
Their actual dimensions are 768x432, whereas the source-video metadata says
1280x720. Pixel checks use actual decoded dimensions, not the source metadata.
`1-1.mp4` failed PyAV decoding with InvalidDataError and was not padded/repaired.

## Perception Diagnosis and Fix

`outputs/detector_prompt_diagnostic/report.json` compares joint, reversed and
single-class prompts on the same proxy/generated images. Joint prompts label
the proxy car as traffic light; single-car prompts find it, while single-traffic-
light prompts can also confidently label it. Changing prompt order does not
resolve the issue. Picking whichever prompt agrees with world state would hide
the detector error, so the default vocabulary is not narrowed to expected objects.

Added explicit contradictory-label detection for nearly identical bboxes (IoU
at least 0.85). Such observations are retained as unmatched/ambiguous evidence,
not accepted as distinct world identities. Added real perception on the compiled
reference before interpreting generated discrepancies. The report now separates
misses already unverified on the reference from misses after reliable reference
detection. This is assessability checking, not a calibrated probability.

`outputs/quality_calibrated_replay` ran the real perception/GPT loop successfully.
It records three conflicting-label groups; all expected categories are unverified
on the crude proxy. It no longer asserts a numeric tree projection error from a
person/tree overlapping detection. The GPT decision explicitly acknowledges
perception ambiguity and leaves world facts unchanged.

## Original RGB Check

Two original RGB JPEGs were visually inspected and four approximate visible
person boxes annotated in `configs/gta_person_annotations.json`.
Joint prompts: TP=4, FP=0, FN=0, mean bbox IoU=0.97839.
Reversed prompts: mean IoU=0.97839. Singleton person: mean IoU=0.98023.
Only persons are annotated; other categories are not labeled negatives.
These are two non-held-out frames, not evidence of general precision/recall,
accurate reconstruction, full occlusion handling or metric camera calibration.

Artifacts: `outputs/gta_rgb_detector_diagnostic/person_evaluation.json`.

## H3 Conditioning

See `geometry_conditioning_assessment.md` for source-level compatibility.
No ControlNet weight is installed. Current Union path targets FL2VA, not the
deployed INT8 Ref2VA reference packing. No semantic-ID support was established.
No large weight downloads or base environment changes were made.

Implemented the already-supported silent-video reference route separately:
`H3Renderer.render(reference_video=...)`. References are timestamp-resampled to
24 fps, short/corrupt/non-monotonic input is rejected; image and video arguments
cannot silently override one another. Selected source indices/times are saved.

`outputs/gta_sampled_reference.mp4` contains 72 frames built from 15 original
5fps sampled RGBs by previous-frame hold; it is explicitly NOT native 24fps
video and is not measured geometry. The provided manifest/pixels are retained.

`outputs/video_reference_smoke/generated.mp4` completed a real three-step H3
video-reference test; subsequent GroundingDINO/SAM2/DINOv2 perception succeeded
on its middle frame with four observations. Three-step smoke is not a quality
claim or a completed image-vs-video 50-step comparison.

## Tests

- Existing 14 contract tests passed.
- Three new perception-quality tests passed.
- Three reference sampling tests passed (native 24fps, 10-to-24fps timestamp
  resampling, rejection of short input).
- Original-data detector diagnostics and true H3 video-reference smoke executed.

## Remaining Input

Need the real GTA episode folder including `.spool/world_state.jsonl` or finalized
Parquet, camera/client observations, frame index/timestamps and `rgb.mp4`.
The manifests reference `/mnt/workspace/wangruiyan/data/ep2/...`; that location
was not supplied on the current server. Without those files, only 2D RGB checks
and inferred-scene consistency can be performed honestly.
