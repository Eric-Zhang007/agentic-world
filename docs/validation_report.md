# Prototype Validation

Server: star-Super-Server. Run date: 2026-10-03 Asia/Shanghai.

## Completed

- Read existing H3 deployment README and real pipeline/quantized recipe.
- Preserved original H3 environment, scripts and weights.
- Inspected all three supplied GTA SQLite databases read-only.
- Installed perception into separate `.venv`, reusing existing torch read-only.
- Deployed public GroundingDINO tiny, SAM2.1 tiny and DINOv2 small.
- Tested user-selected gateway and exact model `gpt-6-astra`.
- Ran 14 deterministic tests: all passed.
- Ran real GPT A/B/C decisions on explicitly labeled controlled observations:
  A ACCEPT, B REGENERATE, C REGENERATE; generated dog never inserted.
- Ran actual 50-step H3 -> sampled frames -> real detector -> SAM masks ->
  DINO embeddings -> Hungarian assignment -> discrepancy -> GPT -> H3 reroll.
- Verified two full outputs each contain 72 frames at 24 fps; seeds 42 and 43.
- Verified masks exist and embeddings are finite 384-dimensional vectors.
- World snapshot before/after the two-attempt episode is identical.
- Replayed with reference-world DINOv2 embeddings in pair costs; the report
  includes real `visual_similarity` (e.g. person frame 0: 0.55316).
- Imported and compiled the supplied inverse scene using its existing CPU
  triangle renderer; 13 objects retained INFERRED provenance and original IDs.

## Real H3 Episode

`outputs/episode_real_50/verification.json`:

| Attempt | Steps | Seed | Renderer subprocess seconds | Unmatched | Missing | Mean center error px | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 00 | 50 | 42 | 216.755 | 0.250 | 0.333 | 50.867 | REGENERATE |
| 01 | 50 | 43 | 216.705 | 0.444 | 0.444 | 25.908 | REGENERATE |

Final status: accepted=false, attempts=2, retry_exhausted=true.
These timings concern a single 480-short-edge reference image, NOT the old
video+audio benchmark. No comparative speedup claim is made.

This verifies the loop and truthful rejection, NOT a successful visual rollout.
The controlled A/B/C fixtures are not claimed as H3-generated scenes.
Actual generated dog hallucination has not been reproduced in a labeled H3 test.
Actual GPT RECONCILE_WORLD on independent original evidence is not yet evaluated;
the evidence-gated update behavior is covered by executed tests.

## Failure Modes

The simple colorful proxy is strongly domain-shifted from the detector's training
images: the visible car is labeled bench/traffic light and the person sometimes
tree. Discrepancy can therefore mix detector error and renderer error.
The observed tree also shifts against expected projection in the actual video.
Ref2VA does not enforce the 3D layout through depth/semantic conditioning.
Unknown categories outside the proposal list and brief events between sampled
frames can be missed. Track matching cannot prove long-term identity.

## Artifact Map

- `outputs/h3_smoke`: three-step renderer test.
- `outputs/contract_cases`: real GPT decisions on controlled A/B/C observations.
- `outputs/episode_real_50`: actual two-attempt 50-step loop and media verification.
- `outputs/replay_real_perception`: real perception on the three-step output.
- `outputs/replay_dino_matching`: reference-DINO-enabled matching replay.
- `outputs/inverse_snapshot.json`: imported provided scene at t=1s.
- `outputs/inverse_compiled`: real source-renderer output and materialized scene.
- `outputs/episode_final_50`: final implementation including GPT planning,
  persistent decision-history retrieval and world-reference DINO features;
  completed and verified: 50 steps, 72 frames, 24 fps, renderer 210.041 seconds,
  unmatched ratio 0.25, missing ratio 0.3333, mean center error 50.870 px.
  GPT returned REGENERATE, zero rerolls configured for this final regression.
  World snapshot and authoritative facts remained identical. Fourteen final
  regression tests and syntax compilation passed after this run.

## Next Three Priorities

1. Obtain original GTA episode JSONL/Parquet plus aligned RGB/camera metadata.
   The supplied ingestion SQLite is not a replacement for these payloads.
2. Calibrate detector/projection on that real paired data, including occlusion,
   partial-world coverage and annotated unmatched/missing cases.
3. Evaluate actual H3 geometric conditioning with a compatible installed
   ControlNet checkpoint, or stronger reference-video conditioning, separately
   from the current prompt+image baseline.
