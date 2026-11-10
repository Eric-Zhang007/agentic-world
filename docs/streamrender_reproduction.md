# StreamRender reproduction on the existing server

Work started October 12, 2026. This is an isolated portability experiment;
passing protocol or component checks is not successful video reproduction.

## Location

Server root: `/mnt/hdd/agentic-world-models/streamrender-repro`.
Source: `Sana/models/streamrender_h3`, pinned to
`62da34b14a53d7289e82fa56692f34e96e089bee`.
Python: `venv/bin/python`, Python 3.11 with existing Torch 2.9.1+cu128.
The new environment reuses installed Comfy dependencies without writing to
the original environment. Missing dependencies are installed in the new venv.
Transformers 4.57.3 and PEFT 0.18.0 are installed locally in this venv,
with tokenizers 0.22.2 and huggingface-hub 0.36.2. The shared Transformers
5.18.0 is incompatible with the upstream positional `get_rope_index` call;
the initial run failed there before any video was generated. Pinning the
upstream versions avoids changing that model code or the original environment.

New student backbone, causal decoder and TAE assets are downloaded through
`hf-mirror.com` into `weights/`. Existing H3 Qwen, audio VAE, tokenizer and
processor are referenced in place. `weights/download_selection.json` records
the model revision and selected files. The indexed safetensors loader avoids
reconstructing a second 66GB backbone file.

## Changes to the public execution profile

- Original 50-layer, two-NFE student weights and streaming logic retained.
- Backbone uses the supplied FSDP placement mechanism over four GPUs instead
  of one full resident model per GPU. Context parallelism remains four-way.
  BF16 autocast wraps backbone compute; coordinate and timestep inputs are
  not indiscriminately cast by FSDP. The second attempt exposed this missing
  context as a Float/BFloat16 linear mismatch before producing a video.
- Existing upstream Qwen safetensors are loaded with explicit exclusion of
  unused upper layers and output head instead of downloading duplicated DCP.
- The same causal ViT24 student decodes using its native PyTorch implementation
  rather than the Blackwell FA4/CUDA-graph implementation.
- Compiled FlexAttention uses 32x32 tiles and two stages on Ada. Default tiles
  exceeded per-SM shared memory (106496 requested, 101376 available). The mask
  and attention operation are unchanged. CUDA tests against masked SDPA at
  head dimension 256 and lengths 257/1024 pass with maximum error 0.00390625.
- No student quantization, layer removal, sparse approximation or step reduction.
- Packed SDPA adds an explicit batch dimension so PyTorch can use fused
  attention instead of materializing long attention matrices. Packed sequence
  boundaries, bottom-right causal/window masks and scale remain unchanged.
  Tests cover unequal query/key lengths and force the CUDA Flash backend on
  a 4096-query/8192-key sequence (12 heads, head dimension 256).
- Offline replay uses the official sample at original output resolution. This
  does not reproduce their browser latency or 8xGB200 throughput numbers.

Adapters live in the agentic project, leaving the pinned source unchanged:
`scripts/streamrender_portable.py`, `scripts/run_streamrender_portable.py`.

## Component verification

- All 535 backbone state keys match the released index.
- All 902 retained Qwen state keys match existing upstream checkpoint keys.
- Actual four-GPU causal-block parameter sharding and two MLP forwards pass.
  This small test is not a full attention/model inference test.
- GPU packed SDPA fallback matches separate-sequence SDPA within tested BF16
  tolerance (maximum absolute difference 0.0078125).
- Upstream standalone causal decoder CPU equivalence test passes.
- Default `B10_e-track-3_bluehour` input has 1450 semantic frames, 24fps,
  1344x768; its reference image is 3584x2048. The exact official prompt is used.

## Launch

After assets and `runtime.portable.json` / `assets.portable.json` are prepared:

```bash
cd /home/star/agentic_world
OMP_NUM_THREADS=4 FLEX_FLASH_ATTN_AVAILABLE=0 FLASH_ATTN_4_AVAILABLE=0 \
  /mnt/hdd/agentic-world-models/streamrender-repro/venv/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/run_streamrender_portable.py \
  --source /mnt/hdd/agentic-world-models/streamrender-repro/Sana/models/streamrender_h3 \
  --config /mnt/hdd/agentic-world-models/streamrender-repro/runtime.portable.json \
  --assets /mnt/hdd/agentic-world-models/streamrender-repro/assets.portable.json \
  --video /mnt/hdd/agentic-world-models/streamrender-repro/media/examples/media/videos/7622f0459a694082db1f.mp4 \
  --output /mnt/hdd/agentic-world-models/streamrender-repro/outputs/official_replay_new \
  --rounds 12
```

Use a fresh output directory per run. Generated chunks, timings and last-frame
images are recorded. Check inference logs and actual outputs before calling a
run successful. A partial MP4 after a failure is not proof of a complete run.

After a successful process exit, run `scripts/inspect_streamrender_output.py`
with the output directory. It verifies decoded frame counts against chunk
metrics and writes `inspection.json` and `contact_sheet.jpg`. These numerical
checks do not establish perceptual quality or cross-game generalization.

## October 15 continuation

The fourth attempt produced a valid 26-frame first chunk but failed on the
second chunk in the original three-dimensional SDPA fallback with CUDA OOM.
It is a partial run, not a successful reproduction. The fifth attempt uses
four-dimensional SDPA, the exact upstream torch nearest-neighbor semantic
resize, and a final decoder flush. It requests 12 streaming rounds at the
original resolution in `outputs/official_v5_20261015`.

The fifth run exited successfully. Decoding and frame accounting passed:
124 output frames = 124 consumed semantic frames, including four flushed
tail frames, 1344x768 at 24 fps (5.1667 seconds), all 12 rounds at two NFEs.
The local copy is `outputs/streamrender_repro_20261015/generated.mp4`.
`contact_sheet.jpg`, `inspection.json`, `timings.jsonl`, `completion.json`
and `run.json` accompany it. Server log: `inference_v5.log` under the
reproduction root.

Loading took 541.27 seconds. Summed chunk RGB-ready wall times were 210.52
seconds, excluding loading, conditioning setup, video IO and the final flush.
The first chunk took 57.77 seconds; the median of subsequent chunks was
13.20 seconds. This four-card FSDP/PyTorch deployment is not real-time.
Its speed is not a matched benchmark of the public GB200 deployment.

Visual inspection of eight sampled frames found continued forward motion,
a broadly stable vehicle identity and no obvious chunk breaks. Signs and
license-plate text remain unstable. This is a successful official racing
sample portability run, not pixel-exact equivalence to the FA4 implementation
and not evidence of GTA or Black Myth generalization.

The final runner writes its completion marker after MP4 close; this ordering
was corrected while v5 was already loaded. V5 used the earlier marker order,
but its exit code was zero and the finalized MP4 was decoded successfully.
The strengthened inspector independently checks all reference/output counts.
