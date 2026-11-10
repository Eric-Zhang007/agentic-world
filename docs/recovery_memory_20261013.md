# Recovery and task-directed appearance memory

Server recovered on October 13, 2026. Four GPUs passed a fresh CUDA matrix
operation in the existing Comfy environment (torch 2.9.1+cu128). The existing
proxy was listening and the configured agent returned a valid JSON response.
No torch/CUDA packages or checkpoints were replaced.

## Real image-memory consumer change

The previous six-image retrieval prioritized all actors and held props before
scenery. Repeated character views could exclude every architectural landmark.
The agent now queries the memory catalog for the current task before loading
crops. It can request persistent object IDs and intended views, including
landmarks that will reappear later in the planned camera path. Each requested
object gets one image before second views are allocated. Legacy non-task
inspection remains available; normal planning, rendering and acceptance use
task-directed retrieval.

Missing objects, missing views and the selector's stated evidence limitations
are carried into planning text, render contracts and acceptance payloads.
Identity fallback is explicitly not proof of an unseen side. The catalog and
retrieval share eligibility: original anchors, or accepted identity references.
Objects without eligible images remain visible as evidence gaps.

`scripts/query_appearance_memory.py` opens its source SQLite read-only, copies
it into memory, runs the real agent query and asserts no snapshot changes.
It does not invoke H3 or commit world edits. Example:

```bash
sh scripts/with_proxy.sh .venv/bin/python scripts/query_appearance_memory.py \
  --world outputs/appearance_memory_20261012/world.sqlite \
  --output outputs/identity_repair_20261013/task_memory_probe_v2 \
  --goal 'Return to the temple with a side camera viewpoint; report missing evidence.'
```

The first real query retrieved player, player_staff, tiger_guard,
right_relief_wall, central_sentinel_plinth and rear_terrace. The crop sheet was
manually inspected. This demonstrates query wiring, not successful video recall.
The source world remains unchanged. H3 still receives only its selected inputs;
agent memory images are not automatically additional H3 references.

## Recovery experiments

- `outputs/identity_repair_20261013/reboot_description`: replay the earlier failed
  prompt with accepted appearance descriptions appended; GPU 0.
- `outputs/identity_repair_20261013/reboot_live`: replay the saved real agent
  compilation with its selected descriptions; GPU 1.

Both retain the pre-outage seed 102, 40 steps, 124 frames, 1024x768, depth
strength 1 and cache disabled. These are diagnostic outputs, not world commits.
Their saved prompts predate task-directed retrieval, so they do NOT measure the
new retrieval strategy. Earlier pre-reboot failures produced no valid videos.
Each new output must pass visual identity and static-world review before any
state can be accepted or used to plan a continuation.

## Verification

56 focused server tests pass: memory capture/retrieval, reference consumers,
camera checks and video reconciliation. New cases cover relevant landmark
retention, requested back views, explicit missing views, missing landmarks,
catalog eligibility and a read-only query. Local memory-only tests pass; full
camera tests use server PyAV rather than installing another local video stack.

## Remaining limits

Metadata selection cannot create missing views. Current scenery crops include
foreground occluders; they are reasoning evidence, not clean scene plates.
Some object descriptions from multiple captures can still conflict, and capture
admission does not independently compare every proposed detail to the original
identity anchor. These remain limitations, not solved by a retrieval query.
Four-region preparation assets are synced, but no new world regions, portal
events, dragon proxy or progression were committed by this work. User priority:
validate consistency first, then expand the demo.

## Completed recovery results

Both replays completed with return code 0 and released GPU memory. Wall time:
`reboot_description` 1095.02 seconds; `reboot_live` 1049.81 seconds, including
model loading and video decode. Manual chronological sheets and separate GPT
camera audits agree that the recognizable armored identity was retained in
these two examples. Both audits support camera motion, identity and
occlusion/reappearance, but reject static-world consistency. Previously visible
carved plinth and relief masonry simplify/change through the orbit. Overall
FAIL for both; neither output is an accepted world continuation.

The description-only replay changes the prompt while retaining the recorded
seed, conditions and generation settings. This supports the importance of
identity descriptions for this case, not a universal fix. The local synchronized
`outputs/identity_repair_20261013/identity_repair_comparison.mp4` includes the
previous failure and both new videos.

The live task-memory runner `task_memory_orbit` failed with proxy SSL EOF during
condition review, before H3 generation. It preserves a separate branch and
does not advance accepted history. `compile_saved_rollout_prompt.py` provides
a read-only diagnostic path: validate exact scene equality and deterministic
plan replay, select task memory, compile a new prompt, and retain source paths.
It does not pretend to resume or commit the interrupted experiment.

The v2 real memory query reports missing side views and unmemorized temple
landmarks explicitly. Requested unknown/front references are not side evidence.

## Scene-memory and strength comparison

The saved plan compiled successfully through the real agent in `task_prompt`.
It selected three actors/props and three static landmarks. Its final prompt
contains the accepted carved plinth, relief wall and terrace descriptions;
the H3 image input is still the same accepted boundary only.

`task_memory_depth1` and `task_memory_depth05` replay this exact prompt at
strength 1.0 and 0.5 on separate GPUs. Direct comparison of the saved command
arrays verifies that ONLY output path and strength differ. Both keep seed 102,
40 steps and the exact existing depth sequence. The override is experimental,
not a new default. Additional tests protect this command identity and saved
historical-prefix semantics; the saved-plan compiler reads prefix_frames from
alignment.json instead of treating every continuation as a single-frame start.

The original planned whitebox versus the first completed real replay is at
`reboot_live/planned_comparison.mp4` locally and on the server. It is explicitly
PLAN versus ACTUAL, not a reconciled-world comparison and not an accepted run.

## Final scene-memory results

| Experiment | Wall seconds | Identity | Camera | Static world | Overall |
| --- | ---: | --- | --- | --- | --- |
| task_memory_depth1 | 772.48 | CONTRADICTED | CONTRADICTED | CONTRADICTED | FAIL |
| task_memory_depth05 | 746.04 | SUPPORTED | CONTRADICTED | UNCERTAIN | FAIL |

Both actual videos completed and were inspected in chronological sheets, then
audited by the same separate camera-review procedure. At strength 1.0 an
additional bare proxy-like humanoid holding a staff emerges while the armored
player remains present. At 0.5, recognizable appearance persists but the
requested orbit is replaced by mostly lateral image rearrangement; familiar
masonry detail is not proof of a rigid 3D world. Do not accept either.

The local `task_memory_strength_comparison.mp4` aligns the two actual videos.
All four new videos and their JSON audits are downloaded under
`outputs/identity_repair_20261013/`. All generation/review processes completed,
GPUs returned to 15 MiB each, no candidate video was committed. The failed live
branch remains revision 8, time 31.625 seconds, with its accepted prefix intact.

Verification totals: 56 memory/camera/reconciliation tests, 57 prepared-plan,
motion/world/action tests, and 3 saved-prompt/ablation tests passed on the server.
These counts are code verification, not video acceptance.

Important confound: task retrieval's six images cover six objects, so each
object gets one reference; the earlier identity-only replay had two descriptions
per actor/prop. Therefore the difference between these experiment families
cannot be attributed solely to adding architectural words. The strength pair
itself DOES isolate strength. Next work should separate stable identity text
anchors from per-shot image allocation, retain reference provenance, and test
representation/control alignment at known architectural boundaries. Do not
declare success by lowering the requested camera excursion or by accepting
an unrelated new humanoid as the planned player's motion.

## Stable identity anchor replay

Text anchor selection now has its own `text_object_ids`, independent of image
slots. Zero-image retrieval can carry object-bound historical descriptions.
Invalid and duplicate IDs are rejected; unseen views remain missing evidence.
Server verification: 63 focused memory, camera, reconciliation, saved-prompt
and ablation tests passed. This does not establish visual success.

`stable_anchor_depth1` completed a real 124-frame H3 replay at the same seed,
depth and strength. Only the recorded identity suffix was replaced, preserving
the existing prompt prefix and object selection. The separate camera audit in
`review_retry1/review.json` returned FAIL: camera and identity contradicted,
static world uncertain. The initial audit failed due to a proxy disconnect;
its incomplete directory is retained. No candidate was accepted or committed.

Chronological review confirms simultaneous armored player and thin proxy-like
staff bearer, followed by departure of the original player. Fixed descriptions
alone therefore did not resolve this failure. The synchronized local artifact
is `stable_anchor_depth1/planned_comparison.mp4`.

Visual comparison of proxy start and accepted first frame shows different body
facing, staff orientation and architectural silhouettes. This supports testing
condition-to-boundary alignment next, but is not an isolated causal finding.
Do not infer that a new character is a legitimate world entity or weaken the
camera acceptance requirement. Historical anchors also remain baselines, not
a completed mechanism for real accepted changes of costume or damage.
