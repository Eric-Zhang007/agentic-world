# Paired Memory in the Pi World Loop

## Inputs

The current default is the epoch-15 LoRA on the existing H3 Ref2VA base. Depth
ControlNet is off. Native whitebox video remains an input, including its camera
trajectory and coarse object motion.

With memory, input numbering is owned by the renderer, not agent-written prose:

| Input | Runtime meaning |
| --- | --- |
| Picture 1 | One horizontal historical RGB-left / whitebox-right pair |
| Picture 2 | Current accepted RGB endpoint, also used as the start guide |
| Video 1 | Compiled future whitebox trajectory |
| Output | Generated RGB video |

Without memory, current RGB is Picture 1 and there is no historical pair.
This is the explicit ablation, not an additional world commit.

## Relation to the Training Prompt

We follow the supplied training screenshot's history + current first frame +
whitebox-video structure and its object_definitions, summary,
attention_analysis, detailed_description, overall_soundscape and
non_diegetic_music sections. We have not inspected the actual training
dataloader, image preprocessing or LoRA alpha configuration. Pair orientation
is RGB-left / whitebox-right and must be confirmed against that implementation.

The runtime adds action/camera intent, historical-vs-current-state semantics,
and an explicit rule against restoring old positions or ownership. Intermediate
motion may be naturally enriched while respecting the coarse trajectory.
Moving clip boundaries need not stop. Runtime has no target RGB video: that is
training supervision. It uses one retrieved pair, not arbitrary individual
object crops or the whole memory catalog.

## World Tools

1. Inspect and approve a candidate scene, then initialize an independent DB.
2. memory_capture(db, output, initial_rgb) records the initialization pair.
3. plan and compile produce a checked future trajectory.
4. render requires db, checks scene/revision/time and current RGB pixels,
   retrieves a pair near the planned endpoint camera, and records selection.
5. inspect_video exposes sampled evidence to Pi/GPT.
6. commit first reconciles the actual accepted trajectory and endpoint. Only
   then does capture pair the decoded video endpoint with that corrected scene.

The initial/accepted whitebox is inferred coarse geometry, not ground truth.
Capture checks that the recorded last RGB really is the decoded final frame.
Registration is revision-idempotent. Capture failure leaves the accepted world
commit intact and returns CAPTURE_FAILED with an explicit retry operation.
Previously accepted history is never replaced by a later planned endpoint.

The checked world render disallows prefix-video and HUD-rewrite overrides of
the accepted first frame. Legacy low-level renderer APIs remain available for
isolated experiments. They are not a replacement for the checked world tools.

## Retrieval and Limits

The first implementation ranks historical camera position and viewing direction
against the planned endpoint. It records both distances, scores and provenance.
It does not yet solve surface visibility, occlusion, or per-object state-aware
reference selection. An endpoint-near camera can still face altered objects.
Actual memory effectiveness must be established with later-frame visual evidence
and a fixed-condition no-history comparison, not inferred from correct wiring.

geometry_propose reuses VGGT, perception and the asset compiler. The returned
scene is a proposal: incomplete surfaces and unknown metric scale still need
inspection. It does not overwrite an accepted world or auto-align a new chunk.

## Verification

Run local deterministic tests:

```powershell
python -m unittest scripts.test_memory_renderer scripts.test_comfy_renderer scripts.test_paired_history scripts.test_geometry_pipeline -v
python harness/pi/test_memory_gates.py
```

The bounded real experiment is specified in configs/memory_return_task.txt.
Its output directory is outputs/paired_memory_return_20261104. Consult its
actual trial_summary and video reviews for results; API tests are not a video
quality or long-horizon consistency result.

### November 4 Real Run

Both epoch-15 outbound clips completed: paired history and no history, with
identical seed/action/whitebox/first RGB and depth disabled. Their appearance and
camera motion are broadly similar, but both fail the same foreground pole
placement. Pi rejected the outbound and kept revision/time at zero; return was
not executed. See outputs/paired_memory_return_20261104/REVIEW.md and its
comparison/comparison.mp4. A historical initial RGB redundant with the current
first frame is not a long-term revisit test.

A separate old accepted-video replay verified real endpoint capture on an
isolated fork. Fresh VGGT/perception/compiler/preview execution took 27.82s,
but its road support was incomplete. Ambiguous detector category phrases no
longer fall through into arbitrary solid boxes. They remain review candidates;
this fix does not invent missing navigable terrain.
