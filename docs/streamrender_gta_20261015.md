# StreamRender GTA garage zero-shot probe

## Result

Execution succeeded; scene rendering failed the requested identity criteria.
The released racing student retained the red/blue proxy figures rather than
rendering the two clothed GTA characters. Parts of the workshop appearance
transferred, with coarse camera/motion following, but background geometry,
sharpness and subject identity were unstable. Around the sampled 2.04-second
frame, an extra textured human-like ghost appears beside the proxy people.
No racing car or outdoor racetrack is evident in the six inspected frames.

This is one scene and one seed, not a measured generalization success rate.
It demonstrates that direct use of our existing GTA character proxy is not
adequate with this checkpoint. It does not establish failure for GTA driving,
or isolate fine-tuning, distillation and input-domain mismatch as causes.

## Inputs and configuration

- Existing data: `/home/star/agentic_world/data/gta_sampled_segment`.
- Proxy: `render_color/model.mp4`, 72 frames, 12fps, six seconds, shaded
  gray environment with red player and blue NPC. Resampled to 24fps with
  previous-frame hold; no invented intermediate motion.
- Reference: corresponding `frames/000.jpg`. Lower-left minimap filled with
  OpenCV Telea. Original, edited reference and invalid mask are preserved.
- Prompt explicitly describes two humans, reference clothing, indoor workshop,
  red/blue identity labels and no racing cars. Full prompt saved with output.
- Same pinned model/runtime and seed as the successful official racing probe:
  50 layers, BF16 four-GPU FSDP, two NFEs per chunk, native causal decoder,
  original 1344x768 output resolution. No training or weight edits.
- 12 chunks, 124 decoded frames, 24fps, 5.1667 seconds. Frame totals match
  consumed conditions including final decoder flush. Process exit code zero.
- Summed chunk RGB-ready wall time: 235.79 seconds; subsequent-chunk median:
  13.91 seconds. Excludes loading, initial conditioning, IO and final flush.

## Confounds

The old proxy visibly misaligns with the RGB scene in framing, scale, geometry
and later camera orientation. It is shaded geometry rather than the flat
semantic palette used by the official example. The published palette has no
human category; red/blue labels normally denote cars. We reinterpret them in
the prompt, which is an out-of-domain intervention, not supported class mapping.
The reference and output aspect ratios also differ slightly (16:9 vs 7:4).
These mismatches prevent attributing all failures to loss of model generality.

## Artifacts

Server inputs: `/mnt/hdd/agentic-world-models/streamrender-repro/gta_garage_20261015`.
Server outputs: `/mnt/hdd/agentic-world-models/streamrender-repro/outputs/gta_garage_20261015`.
Server log: `/mnt/hdd/agentic-world-models/streamrender-repro/inference_gta_20261015.log`.
Local outputs: `outputs/streamrender_gta_20261015/`.

`generated.mp4` is the raw model output. `comparison.mp4` and
`comparison_sheet.jpg` show original GTA / approximate proxy / generated video.
Original RGB comes from existing 5fps sampled frames held at their timestamps;
it is not full-rate source footage. `inspection.json`, `timings.jsonl`,
`run.json`, `completion.json`, `input_manifest.json` and `prompt.txt` preserve
the execution and input evidence. Visual findings are human/assistant inspection
of sampled frames, not detector metrics or verified contact/causal evaluation.

Preparation and inspection entrypoints:
`scripts/prepare_streamrender_gta.py`, `scripts/run_streamrender_portable.py`,
`scripts/inspect_streamrender_output.py`, `scripts/compare_streamrender_gta.py`.

## Most useful next controls

1. GTA driving with exact official palette and an aligned initial reference,
   testing the smallest shift from racing training.
2. This garage scene with exact flat semantic proxy colors and improved camera
   alignment, holding checkpoint, seed and prompt fixed.
3. Compare the same aligned inputs with the original general H3 renderer to
   separate conditioning defects from specialized-student limitations.
