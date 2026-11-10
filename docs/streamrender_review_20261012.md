# StreamRender-H3 audit, 2026-10-12

## Verdict and evidence boundary

StreamRender-H3 substantially overlaps the broad idea of an explicit code world
plus generative rendering. That combination alone is no longer a credible
distinction for our project. Its released checkpoint is demonstrably a
racing-specialized deployment; GTA open-world play and articulated fantasy
combat generalization are not established by the published evidence.

This audit ran release CPU tests and interface probes, inspected the actual
checkpoint tensor header, and read the project page, model card and runtime.
It DID NOT generate GTA or Black Myth video using their checkpoint. We therefore
have no measured out-of-domain success rate and no grounds to call the checkpoint
incapable of those tasks. The local hardware cannot run the unmodified public
resident BF16 configuration. No model weights were downloaded or existing
inference environments changed.

Pinned code commit: `62da34b14a53d7289e82fa56692f34e96e089bee`.
Root README dates the release to October 11, 2026. This is a new release; absence
of independent evaluations should not be treated as a negative evaluation.

## Sources

Official source URLs, retrieved directly from the server through its proxy:

```text
https://github.com/NVlabs/Sana/tree/sol-engine/models/streamrender_h3
https://nvlabs.github.io/Sana/Sol-Engine/StreamRender-H3/
https://huggingface.co/Efficient-Large-Model/StreamRender-H3
```

Server evidence root:
`/home/star/agentic_world/outputs/streamrender_audit_20261012/`.
`project.txt` preserves project-page text, `model_card.txt` the model card,
`source/` selected pinned source files, `tree.json` the commit tree,
`backbone_size.json` the header inspection, and `release_audit.json` test results.
Built-in web retrieval returned no content. Tavily returned irrelevant results;
none of those were used. All substantive external claims below use official
material fetched directly, not search-generated summaries.

## What it actually does

1. A Three.js game engine maintains map geometry, physics and object behavior.
2. It streams flat-color semantic RGB frames at 1344x768 and 24 fps.
3. Qwen encodes a reference image and structured prompt once per session.
4. A trained streaming H3 Ref2VA backbone renders chunks with two denoising
   evaluations, an opening sink and recent generated history.
5. A trained causal decoder publishes images to the browser; player input drives
   the next engine step.

The semantic stream is reference-video conditioning, not the depth ControlNet
interface used by our current Comfy experiments. Depth, instance IDs, articulated
skeletons and event logs are not separate runtime conditioning channels here.
Transport metadata records controls, but `runtime/worker.py` feeds image bytes
to the renderer and logs controls separately. This does not mean the renderer
ignores actions: their consequences appear through engine-produced images.

The model is not simply original H3 with EasyCache enabled. It has streaming
fine-tuning, two-step distribution matching distillation, a changed context mask,
and a distilled causal decoder. Inference includes exact cache/layout/kernel
optimizations as well. These different changes must not be conflated.

Evidence: project sections 2 and 3; `render/pipeline.py`, `runtime/worker.py`,
`configs/runtime.json`, model card.

## Generalization evidence

The project explicitly limits its reported results to racing. Training examples
use AI-driven cars on 39 tracks, a chase camera behind the player's car,
10,000 approximately 15-second paired clips and racing streams for DMD.
There are varied surfaces, lighting, weather and car appearances. Those are
meaningful variations, but are not demonstrations of new gameplay mechanics.

The catalog audit finds 60 requests, 30 distinct semantic videos and 60 pictures.
Group A has 20 examples; group B contains 10 trajectories with four appearance
variants each. A public 20-scene showcase is not a 20-domain benchmark. No
cross-game held-out evaluation or per-task failure rate was identified in the
inspected public material. Curated examples alone do not establish one.

The palette contains 13 classes plus unlabeled: road, road edge, terrain, grass,
vegetation, rock, barrier, building, stands, billboard, sky, hero car and opponent
car. It does not include person, animal, weapon, held object or projectile.
There is no demonstrated instance-specific class vocabulary for interacting
characters. This is a learned-conditioning limitation, not a hard prohibition:
new labels or red/blue actor reinterpretations can still be attempted, but those
would be out-of-distribution probes requiring actual measurement.

Expected difficulty, as hypotheses rather than measured results:

| Test | Shift from released data | Current evidence |
|---|---|---|
| GTA-like chase-camera driving | Vehicle geometry and streets; much existing palette reusable | Most plausible transfer, unmeasured |
| GTA walking, exiting a car, picking up a prop | New articulated classes, ownership and topology changes | Not demonstrated |
| Fantasy character locomotion | New silhouette, gait and appearance-class binding | Not demonstrated |
| Black Myth-style contact, dodge, knockdown | Multi-body articulation, short contacts, occlusion and consequential events | Not demonstrated; strongest shift |
| Camera orbit, first-person, revisiting occluded identities | Viewpoint and long-term visibility changes | Outside stated chase-camera training scope |

It is not valid to conclude that real-time acceleration CAUSED weak
generalization. Racing-only supervision is directly documented. Distillation,
short context and decoder changes might also matter, but attribution requires
matched comparisons with the pre-DMD racing model and the same base H3.
Exact KV reuse or metadata reuse does not by itself imply lost semantics.

One precision issue in the prose: the limitation section broadly describes the
critics as racing-trained, but section 3.1.3 specifies the real critic is frozen
released H3 with no adapter; the fake critic is adapted on generated racing
spans. The accurate statement is that the reported learning/evaluation setting
is racing-specific, not that every teacher component was trained from scratch
or independently fine-tuned on racing.

## Are the claims overstated?

| Claim | Assessment |
|---|---|
| Explicit code state plus streaming neural rendering | Supported by public architecture and runnable protocol code |
| Racing specialization and limited current scenario | Authors acknowledge it explicitly; not a hidden limitation |
| Other engines can supply semantic frames | Reasonable interface claim, not evidence of zero-shot model generalization |
| Real-time performance | Reported for 8 GB200; not reproduced on our hardware |
| Approximately 218ms | Diffusion-stage warm streaming-step timing, not complete input-to-display latency |
| 3.67x speedup | Against their two-NFE PyTorch runtime baseline, not a like-for-like speed comparison with our 40-step pipeline |
| Map revisits and exact collisions | Engine properties; they do not prove generated pixels preserve all geometry and contacts |
| Photorealistic worlds in general | Broader aspiration than the present cross-domain evidence establishes |

The browser acceptance script checks readiness, chunk progress, output image
dimensions and browser errors. It does not enforce visual control accuracy,
collision contact, identity persistence, or an end-to-end latency percentile.
Its successful run would validate plumbing, not all those research claims.
We ran the CPU HTTP test, not this browser/H3 test.

Thus, calling the project fake or claiming it cannot generalize is unwarranted.
Interpreting its architectural portability as proven arbitrary-game zero-shot
rendering would overread the results. Their own limitations are relatively clear.

## Actual tests and deployment findings

Run from our project:

```bash
.venv/bin/python scripts/audit_streamrender_release.py \
  --source outputs/streamrender_audit_20261012/source \
  --output outputs/streamrender_audit_20261012/release_audit.json
```

- Upstream protocol: 4 tests pass.
- Upstream HTTP: 1 test passes.
- Upstream examples/config relocation: 2 tests pass.
- Session geometry: 1344x768 at 24 fps accepted; 1024x768 and 832x480 rejected.
- Checkpoint header: 66,211,541,504 BF16 bytes plus 68,888,640 F32 bytes,
  about 61.73 GiB of backbone tensors alone.
- Server: four RTX 4090 cards reporting 49,140 MiB each; PCIe/NUMA interconnect,
  no NVLink in `nvidia-smi topo -m`.
- Public worker accepts 4 or 8 GPU ranks. However it uses context parallelism
  with resident full models, not automatic parameter sharding across those cards.
  Four-card support is not evidence that a 48GB card can hold this checkpoint.
- Root filesystem has about 47GB free, but `/mnt/hdd` has about 10TB free.
  Storage can be solved without removing any old experiments. No blanket
  download of the 206.87GB bundle was attempted.
- The published reconstruction needs another 66.28GB. Qwen's two storage formats
  duplicate the same model; selective acquisition could reduce download needs.
- Tested dependency stack is Torch 2.9.1+cu128 and FA4/CUTLASS components. Their
  Blackwell profile is not a validated drop-in 4090 configuration.

A faithful inference trial needs suitable high-memory GPUs, or an explicitly
separate portability exercise (offload/sharding/quantization plus compatible
kernels). Quantizing only the student and comparing it to an unquantized teacher
would confound a generalization comparison. Lowering resolution alone does not
solve resident parameter memory.

## Implications for our project

The shared architecture is substantial: explicit persistent world, a low-detail
render representation, reference-conditioned video and long-term consistency
goals. Claiming novelty solely for whitebox-to-video or code-maintained world
state is weak after this release. Our current pipeline is also much less mature
as a high-throughput renderer, so speed is not a credible present advantage.

The potentially distinct question is WHAT becomes authoritative when the video
does something different from the plan:

- Their runtime: engine state is authoritative; semantic frames drive video;
  generated RGB returns to the player. No video-to-world reconcile/accept tool
  was found in the inspected runtime path.
- Our target: pre-generation state is a plan; accepted visible consequences
  update coarse trajectories, events and endpoint; immutable history constrains
  the next action. Renderer detail can remain uncommitted. Consequential unknowns
  trigger evidence tools, clarification of coarse state, or regeneration.

These semantics have different risks. Engine authority can render an apparent
miss while game logic has already counted a hit. Video-driven reconciliation
can drift or rationalize a hallucination into a fact. Our contribution exists
only if experiments show the latter risk is managed and action/world agreement
improves over the simpler engine-authoritative baseline.

Our current evidence is insufficient to claim superiority: two-segment posture
continuation and a single accepted arrival are not broad generalization. The
latest static-anchor audit also found initial scenery/camera misalignment.
That must be fixed transparently in a new calibration branch before a fair
comparison. It is not evidence that StreamRender would fail the same scene.

## A fair next experiment

Use the same initial frame, scene geometry, semantic stream, action sequence,
resolution and random seeds where comparable. Include an in-domain positive
control before reporting any out-of-domain failure. Resample our existing
controls carefully to their 1344x768 format; never stretch class colors with
bilinear interpolation. Do not compare our depth-only input against their
semantic input as if that isolates the model.

1. Racing chase-camera baseline, then GTA chase-camera driving with unchanged
   label meanings. Measure road following, actor placement and appearance.
2. GTA character/prop trial: walk to a location, occlude and reappear, pick up or
   release an object. Report novel-label vs car-label-reinterpretation separately.
3. Fantasy interaction: approach, dodge behind an obstacle, strike with visible
   contact, then drop/pick up a weapon. Require observable evidence before
   declaring damage or ownership changes. Do not score cinematic attractiveness
   as successful action execution.
4. Continue five consecutive actions per scene. Log actor and camera drift,
   retained identity, contact timing, ownership, missing objects, contradictions,
   completion and retries. Use several scenes/seeds and retain failures.

Separate model and system comparisons:

- Model: native H3 vs released streaming student, matched conditions; a
  pre-DMD model is needed to isolate the cost of two-step distillation.
- System: engine-authoritative updates vs our accept/reconcile loop using the
  SAME renderer. This isolates whether our world-maintenance mechanism helps.
- Skills: GPT sparse-frame review alone vs optional motion/matching/alignment
  evidence, measuring wrong acceptance, correct rejection, edit accuracy and cost.

Our best next three investments are initial coarse scene alignment, consequential
interaction/ownership reconciliation, and a multi-action benchmark with these
ablations. Treat StreamRender as both a strong baseline and a possible future
renderer backend, not a reason to defend an untested broad-gen claim.
