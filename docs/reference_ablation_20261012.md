# Camera appearance regression: single-reference replay

## Question

The camera-orbit trials abruptly replaced the established armored character
with a thin proxy-like body. Earlier moving-character and dolly trials did not
show this particular failure in inspected samples. Treat this as a regression
candidate before attributing it to a general limit on large camera changes.

## Recorded comparison

Source: `outputs/camera_orbit_20261012/two_segment_orbit/segment_00/render/`.
Replay: `outputs/camera_orbit_20261012/single_reference/`.

The replay uses the original recorded inference command, with the same seed
102, 40 steps, 1024x768, 124 frames, depth strength 1.0 over the full sampling
interval, first-frame guide and disabled EasyCache. It directly reads the
original `conditions/first.png` and `conditions/depth.npz`; it does not rebuild
the world, camera path or control video. No world acceptance/commit runs.

Changed inputs:

- Remove six historical object crop images from Ref2VA conditioning.
- Point body-text picture references to the remaining full-scene Picture 1.
- Remove instructions about additional pictures and retain their textual
  identity descriptions without image tags.

This tests the additional-reference mechanism as a package, including the
necessary prompt adaptation. It is not an image-only ablation with identical
text embeddings. Text-only and picture-only effects would require further
controlled runs if this package changes the outcome.

`scripts/run_reference_ablation.py` records the exact new command and prompt,
refuses an existing output directory, then calls the existing inference worker.
No model implementation, checkpoint, ControlNet weights or environment changes
are part of this replay.

## Checks before inference

- Both camera-orbit reference PNGs equal the previous accepted video frame
  `appearance_memory_20261012/segment_00/review/frames/000123.png` pixel for pixel.
- All six removed crop references have accepted generated-video provenance.
- The orbit's initial depth array is identical to the initial depth array in
  `natural_walk_dolly_20261012` and `appearance_memory_20261012`.
- Their initial scene cameras and primary actor geometry agree. The only scene
  entity differences found are the offscreen watchkeeper's position and pose.
- The earlier natural-walk/dolly experiment also used depth strength 1.0 but
  only one reference block. Its inspected samples preserve the armored body;
  it was rejected for insufficiently established traversal, not proxy-like
  identity replacement.
- The inference-worker diff against `run_comfy_control.py.before_memory_20261012`
  adds optional reference images, their tensor encoding and metadata/CLI entry.
  No sampler or depth-weight change appears in that diff.

## Native reference/control audit

A separate read-only review of the installed ComfyUI source found no concrete
reference-count-dependent target-index error. Target video rows are marked
explicitly in `comfy/ldm/minimax/model.py`; `controlnet.py` maps hint tokens using
that mask. Output extraction selects the named video segment. Reference order
in `comfy/model_base.py` agrees with layout construction, and layout is rebuilt
for each sampling run.

There is a real coupling path: ControlNet residuals update reference/text hidden
rows as well as target rows; the native residual application masks audio only.
Zero depth hints on reference rows do not isolate those rows from joint control
computation. This is an implementation observation, not a demonstrated bug or
the established cause of the thin-body failure. Removing references changes
text-encoder inputs, reference latents and their control interaction together.
No native ComfyUI code was modified for this test.

## Reproduction

Run from `/home/star/agentic_world`, using a new output directory:

```bash
.venv/bin/python scripts/run_reference_ablation.py \
  --record outputs/camera_orbit_20261012/two_segment_orbit/segment_00/render/generation_metadata.json \
  --output outputs/camera_orbit_20261012/single_reference_repeat \
  --gpu 3
```

## Result

Real inference completed successfully in 745.74 seconds of wrapper wall time.
Recorded output metadata confirms one reference block, native guide at frame 0,
and the same `ref2va_pruned_int8_convrot` base model. All eleven compared replay
parameter/path checks passed. The original seven-reference run is retained.

| Audit | Seven-reference source | Single-reference replay |
| --- | --- | --- |
| Camera execution | CONTRADICTED as a continuous accepted-scene orbit, despite real parallax | SUPPORTED |
| Character identity | CONTRADICTED | SUPPORTED |
| Static architecture | CONTRADICTED | CONTRADICTED |
| Occlusion/reappearance | SUPPORTED | SUPPORTED |
| Overall | FAIL | FAIL |

Direct chronological inspection agrees with the identity distinction. The
single-reference replay retains substantial armor, skirt panels and leg volume
instead of abruptly becoming the thin proxy-like figure in the source run.
It still performs a substantial viewpoint change; this is not an identity win
caused by freezing the camera. The statue pedestal and decorated stair-side
masonry nevertheless change form, so this clip is not accepted world history.

This is positive evidence against the earlier explanation that large camera
movement alone necessarily causes the thin-character failure. The added
multi-picture memory package materially changes the outcome in this fixed-seed
comparison. It does not identify whether image composition/count, prompt
conditioning or reference/control interactions are responsible. One comparison
does not prove all multi-reference configurations fail or that single-reference
conditioning is generally sufficient.

No production memory retrieval was deleted or permanently disabled; the
one-reference route is isolated for diagnosis. No world database was opened
for writing by this runner. The existing accepted parent world remains the
continuation source. The inference and audit jobs exited and GPU allocations
were released.

Local artifacts under `outputs/camera_orbit_20261012/single_reference/`:

- `reference_comparison.mp4`: synchronized source on the left, replay on the
  right; 124 frames, 24 fps, H.264, 1280x512.
- `generated.mp4`, `samples.jpg`, `compare_012.jpg`, `compare_062.jpg`.
- `metadata.json`, `ablation.json`, `prompt.txt`, `camera_review.json`.

Server artifacts additionally preserve raw inference under `render/`, audit
frames and full review under `camera_review/`, and `exit.json`/generation logs.

The next isolation step is to separate memory-derived text from added image
tokens while holding the orbit and seed fixed, then test a small number of
nonredundant whole-scene/object references. Do not raise ControlNet strength or
write scene-specific prompt exceptions on the basis of this result.

## Follow-up: identical text, different image inputs

Two further real generations reused the single-reference replay's exact prompt,
seed 102, first-frame file and full depth file. Only supplementary images differ:

| Trial | Additional image inputs | Wrapper wall time | Identity audit |
| --- | --- | --- | --- |
| `single_reference` | none | 745.74 s | SUPPORTED |
| `three_crops_same_text` | first player/staff/tiger crop set | 868.38 s | CONTRADICTED |
| `six_crops_same_text` | all six original crops | 824.12 s | CONTRADICTED |

Both added-image variants reproduce the thin-body failure. The six-crop version
briefly preserves the original before transforming by roughly one second; the
three-crop version changes in the first nonzero samples. Both retain substantial
camera movement, and both fail static architecture as well. Timing is from
sparse visible samples, not a precisely measured onset frame.

The added images therefore affect the failure even with identical text: prompt
editing alone cannot explain the earlier improvement. Removing the second set
of near-duplicate views does not solve it either. This is one seed and scene;
it does not show all multi-image H3 conditioning is broken. The exact interaction
between text-encoder image inputs, DiT reference tokens and ControlNet remains
unisolated. No assertion of a native tensor-index bug is supported by the audit.

Inspect `outputs/camera_orbit_20261012/reference_count_comparison.mp4`:
top left is the boundary-only run; top right adds three crops; bottom left adds
six crops. Every panel uses identical prompt text and follows the same planned
path. The bottom-right cell is intentionally empty. The contact sheet at
`input_audit/memory_crops.png` also shows that the broad staff crop contains the
player and background. Its object label does not make it an isolated prop.

Reproduce either variant with `run_reference_ablation.py` and a fresh output:

```bash
# Append this to the earlier replay command for the first crop set:
--appearance-indices 1 2 3
# Or use all recorded additional crops:
--appearance-indices 1 2 3 4 5 6
```

## Consumer separation repair

`retrieve()` still returns the full bounded appearance memory for agent planning,
prompt reasoning and acceptance review. Its provenance and early identity anchors
are unchanged. Accepted memory is no longer automatically treated as appropriate
visual conditioning for the next render.

`compile_prompt()` now distinguishes agent image indices from actual renderer
Picture tags. By default H3 receives the accepted full-frame boundary only.
The agent still sees the memory crops and can use them to write stable identity
descriptions. `renderer_references.json` records the actual selection. Explicit
supplementary reference IDs are supported for separately verified experiments;
unknown, duplicate and unaccepted IDs are rejected and selected images receive
their own consecutive tags. A prompt referring to an unselected image is rejected
before generation. The evidence runner passes only this recorded renderer list.

This removes automatic coupling rather than installing a scene-dependent crop
count, ControlNet threshold or prompt exception. It is not an automatic learned
reference selector. The default path deliberately uses the empirically improved
input arrangement while preserving the memory and explicit multi-image API.

An integrated real test is running at
`outputs/camera_orbit_20261012/memory_consumer_fix`: the original prepared plan
passed deterministic replay validation; GPT saw six memory images and emitted
a new prompt with one actual H3 reference. The complete regenerated depth array
is identical to the source experiment. Final video/audit outcome is recorded
below after completion.

The integrated pre-repair consumer test completed before the final identity-text
preservation patch. It used one H3 reference but GPT's generated prompt omitted
several distinctive memory details; the resulting clip again failed identity
and static-world review. Its unchanged world branch is retained as evidence,
not as a successful fix. The latest patch now requires/records
`appearance_object_ids` and appends the selected accepted identity descriptions
intact to the final render prompt. This patch has 26 focused tests passing; it
still needs one fresh real rollout before being called visually validated.
