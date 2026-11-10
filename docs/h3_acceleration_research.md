# MiniMax H3 acceleration research

Checked: 2026-10-08. Scope: read-only server inspection and primary-source
research. No dependencies installed, weights downloaded, environments changed,
runner edited, or GPU inference launched for this investigation.

## Recommendation

There is no demonstrated universally optimal H3 recipe, and no controlled
benchmark found for our exact pruned Ref2VA + Union 2.0 + GPU configuration.
Use the existing 40-step recipe as a reversible quality baseline. First test
native EasyCache alone, then matching Ref2VA PDD at 8 NFE without EasyCache.
Treat PDD + ControlNet as an unvalidated combination, not a promised speedup.
Investigate kernel/backend improvements separately from approximation methods.

The practical objective is **accepted action rollouts per wall-clock minute**,
not merely low seconds per denoising iteration. A fast run that loses identity,
violates a trajectory, or causes repeated regeneration may be slower overall.

## Live server evidence

The following were checked through `python server_exec.py "..."` on this date:

| Item | Observation |
| --- | --- |
| Comfy source | `/home/star/agentic-world-comfy/ComfyUI` |
| Commit | `95539f56344958339e39b7582a476267d489b0ee` |
| Source modifications | `git status --short` returned empty |
| Python environment | `/home/star/agentic-world-comfy/.venv` |
| torch | pip reports 2.9.1, loaded package location is original H3 environment |
| CUDA runtime | Existing deployment records cu128; current `quant_ops.py` retains the cu130 gate |
| Triton | 3.5.1, from original H3 environment |
| comfy-kitchen | 0.2.35, isolated Comfy environment |
| SageAttention / diffusers | Neither found by this environment's `pip show` |
| Hardware | Four NVIDIA GeForce RTX 4090, each reports 49140 MiB; driver 580.178.04 |
| PDD/Turbo cache search | No name matches to depth 3 in `/mnt/hdd/agentic-world-models`, `/mnt/hdd/MiniMax-H3-Quant`, `/home/star/.cache/huggingface/hub` |

The cache search is bounded, not proof that no weights exist elsewhere.
No `torch.cuda` workload or CUDA kernel benchmark was run. GPU inventory used
`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`.

Existing runner read during inspection: `scripts/run_comfy_control.py`. It sets
Comfy arguments internally to `--disable-dynamic-vram` and
`--use-pytorch-cross-attention`. External CLI flags cannot be assumed to change
its attention/backend choices. It loads the pruned INT8 Ref2VA base, INT8 text
encoder, original video VAE and pruned INT8 Union 2.0 control branch. Kernel,
cache, or compile changes require deliberate integration by the runner owner.

Historical local evidence (2026-10-03, not rerun here):
`docs/comfy_control_trial.md` records 832x480, 73 frames, 24 fps, seed 43,
Euler/simple, CFG 1, video/audio shifts 12/3. Sampling plus decode/save took
281.11 s with control and 222.68 s without. These are not pure denoiser timings
and must not be compared directly with third-party s/it or H200 measurements.

## EasyCache: first low-deployment-cost experiment

Installed source [S1] provides `EasyCacheNode.execute(model, reuse_threshold,
start_percent, end_percent, verbose)`. It clones the model and registers
OUTER_SAMPLE, CALC_COND_BATCH and DIFFUSION_MODEL wrappers. The installed defaults
are 0.2, 0.15 and 0.95; the node is marked experimental.

Mechanism: estimate output change from subsampled latent change, accumulate
that estimate, and reuse a cached output-minus-input residual while below the
threshold. This skips a full model evaluation rather than reducing nominal
sampler steps. The outer wrapper resets cache state after sampling.

The installed H3 forward [S2] converts carried audio before entering the model
wrappers and converts the output back afterwards. EasyCache explicitly handles
video/audio lists. Thus the relevant audio-carry implementation exists at our
commit; this is source compatibility evidence, not a successful runtime test.

Suggested integration snippet (for the runner owner; not executed here):

```python
from comfy_extras.nodes_easycache import EasyCacheNode

# Apply to the final configured model before invoking the sampler.
cached_model = EasyCacheNode.execute(
    model=model,
    reuse_threshold=0.10,
    start_percent=0.15,
    end_percent=0.85,
    verbose=True,
)[0]
```

Suggested trials: bypass, then thresholds 0.05, 0.10, 0.20 with the same 40-step
schedule and 0.15-0.85 window. These are conservative experiment choices, not
published optimal H3 settings. Start/end are mapped through percent-to-sigma;
do not assume that they are exact ordinal step fractions after sigma shifting.

Record skipped evaluations and actual denoiser/ControlNet call counts. Because
H3's control work is inside the wrapped model computation, skipped evaluations
can also avoid fresh control evaluation. Check action timing and trajectory
adherence, not just frame similarity. The built-in printed speedup is based on
step counts, not measured end-to-end wall time. No skipping can legitimately
mean little/no benefit; it does not prove integration failure.

Do not reuse cached residuals across actions, changed references, conditions,
camera plans or episodes. Do not initially combine EasyCache with a distill.

## PDD and Turbo are separate recipes

### Alibaba PAI PDD

Publisher [S3] provides separate FL2VA and Ref2VA 8-step BF16 rank-64 adapters.
Its examples use `apply_pdd_lora` and require Diffusers >=0.40.0. This is not
available in our inspected environment, and installing a separate backend is
not necessary merely to evaluate the Comfy route.

The community implementation [S4] explains that the checkpoint includes both
trunk LoRA and step-conditioned output-head banks. It requires Euler, its own
SIGMAS output, CFG 1, shifts 12/3, and strength 1 for both trunk and heads.
It targets Comfy >=0.33.0 and explicitly rejects stacking EasyCache/blockcache
or other distillation LoRAs. Begin at 8 NFE; 4 NFE is a separate quality tradeoff.
Increasing this adapter to 40 steps is not a quality fallback: disable the
adapter and restore the original schedule instead.

Pruned AdaLN is the critical compatibility issue. A dense adapter expects a
2688-wide input but the pruned curve representation has width 8. [S4] implements
curve-basis rebasing including a DC bias correction. Merely ignoring missing
AdaLN keys is not a valid successful load. It also checks FL2VA/Ref2VA pairing.

Important nuance: the installed Comfy `FinalLayer.forward` already includes a
native multi-head PDD bank path and requires `sample_sigmas` when that bank is
present [S2]. We found no PDD-specific node in installed `comfy_extras`, nor
evidence from this inspection that a generic loader completely converts the
publisher's checkpoint. Do not install two competing head patches. Pick either
a verified native converted format or a pinned complete adapter implementation,
and inspect module coverage before generation.

Recommended PDD evaluation gate:

1. Pin the adapter source and correct Ref2VA weight revision; inspect existing
   caches again before any future download through `hf-mirror.com`.
2. Validate all trunk/AdaLN/head keys, pruned basis and sampler schedule on CPU.
3. Run an 8-NFE reference-only case before enabling Union 2.0.
4. Repeat with the exact same world-derived condition; instrument residuals,
   head selection, real NFE and output quality.
5. Only then test 4 NFE. Do not combine with EasyCache in this experiment.

The adapter author's offload benchmark [S4] indicates pre-baking the trunk helps
when runtime patches are reapplied to offloaded modules, but does not materially
help a fully resident model. That is an H200/forced-offload observation, not our
server result. Baking creates a new large quantized checkpoint and is not the
first choice here. We performed no baking or downloads.

### LightX2V / ModelTC Turbo

Publisher [S5] offers multiple different FL2VA and Ref2VA variants. Its Ref2VA
4-step v0.1 uses shifts 12/3 and recommends reference resize mode `match` (target
pixel budget, no forced 2048-pixel reference). Some FL2VA 768p variants instead
use video/audio shifts 6/3. Do not transfer settings between these checkpoints.

A separate Larryvrh Turbo implementation [S6] advertises 4-8 steps and pruned
support via reconstructed time conditioning. Its README warns that its v4
4-step large-motion results can smear; 6-8 steps are suggested for that family.
This is particularly relevant to combat actions, but is an author report, not
a benchmark of our action scenarios. Do not conflate this project's v4 weights
with ModelTC's numbered releases or the PDD family.

Quantized/pruned compatibility and Union control adherence remain local test
requirements for every Turbo family. File naming or a successful LoRA load
alone is insufficient evidence.

## Quantization, attention, compilation and serving

| Option | Source evidence and current recommendation |
| --- | --- |
| Pruned AdaLN | MiniMax describes precomputable AdaLN modulation [S7]. Our pruned base already exploits this class of optimization. Preserve variant compatibility with control and LoRA weights. |
| INT8 ConvRot | Comfy's model publisher recommends INT8 ConvRot when torch cu130 is usable, with FP8 scaled as fallback [S8]. Installed `quant_ops.py` disables kitchen CUDA below 13.0 [S9]. Current successful inference is not proof of optimized INT8 kernels. |
| Triton kitchen backend | Installed CLI supports `--enable-triton-backend`; it is otherwise disabled [S9]. Triton 3.5.1 exists, but operator coverage, numerical correctness and speed are untested. A no-new-weight candidate for a separate A/B. |
| Separate cu130 torch | Potential route to the publisher's recommended kernels. Only in a new complete isolated environment, with driver/runtime/operator tests. Do not upgrade the shared H3 torch or assume its `.pth` overlay makes isolation automatic. No installation done. |
| SageAttention2/2++ | Primary project [S10] supports Ada and documents CUDA >=12.8 for 2++; its advertised attention-kernel speedups are not H3 end-to-end speedups. Current Comfy supports `--use-sage-attention`, but package is absent and runner forces PyTorch attention. Test after installing only into an isolated environment. |
| SageAttention3 | Blackwell-specific implementation [S10]; not the appropriate default recommendation for these RTX 4090s. |
| SDPA / FlashAttention | Installed H3 calls Comfy `optimized_attention` [S2]. Keep current SDPA as the reference. Alternative attention kernels require matching masks/layout/dtype and quality tests; no hardware-universal winner asserted. |
| torch.compile | Installed experimental `TorchCompileModel` exposes `inductor` and `cudagraphs`, clones with dynamic behavior disabled [S11]. Warmup, graph breaks, dynamic reference lengths, CPU offload and head/cache patches can dominate. Measure cold and warm separately. |
| Native sparse attention | MiniMax README still states initial release uses full attention and separate sparse release is planned [S7]. Installed H3 has a VSA gate unused by dense forward [S2]; a gate tensor is not evidence sparse acceleration is enabled. |
| Multi-GPU serving | MiniMax lists SGLang, vLLM, Diffusers and Comfy and shows 4-GPU Ulysses serving [S7]. This does not establish Union 2.0 feature parity or a benefit on this server's interconnect. Do not switch the working control backend solely on that example. |

The existing CLI also has a Comfy compiler switch; that mechanism is not the
same thing as explicitly patching `TorchCompileModel`. Avoid attributing any
warmup gain to torch.compile without logging the actual active path.

Lower resolution, fewer frames, smaller reference token budgets, encoder reuse
and avoiding repeated model loads are additional experiment axes. They change
different portions of total latency. Keep references/precomputed encodings
strictly scoped to unchanged inputs; never use stale conditioning after an
agent world edit. Preserve staged model residency so perception and H3 do not
fight for memory. A persistent worker may amortize loading, but is outside this
read-only task and should not keep all perception models resident.

## Benchmark contract and creative freedom

Use one fixed same-segment input/reference/control/prompt and seeds 43, 44, 45.
Compare each acceleration against its own same-checkpoint baseline. Keep audio
mode, canvas, reference sizing, temporal length and control strength identical.

Record cold load/text encode/control encode, denoiser, decode/save and total
wall seconds; actual NFE; cache skips; each control block's calls; peak allocated
and reserved VRAM; output frame count/fps; model/adapter revisions and complete
settings. Warm timing must not hide first-run compilation or loading overhead.

Score identity, missing/unmatched objects, camera-conditioned projection error,
action order/completion/contact plausibility and frame-boundary continuity.
Do not judge only by PSNR, Canny agreement or an attractive contact sheet.
Compare at equal output duration and include retries in final throughput.

Keep the world constraint deliberately coarse: stable entity identity, rough
location/extent, relative depth, intended trajectory, camera plan and action
milestones. Let the generator supply texture, lighting, muscle/joint detail,
cloth, incidental particles and micro-motion unless they affect an explicit
goal. Do not force a coarse proxy silhouette or absent geometry into dense
pixel-accurate truth. Mark unknown geometry and HUD regions as unassessable;
black condition pixels are not automatically a model-supported unknown mask.
Evaluate tighter vs looser control separately from acceleration to avoid
confounding speed experiments with reduced creative freedom.

Suggested order: 40-step baseline -> EasyCache-only sweep -> PDD 8-NFE
reference-only -> PDD 8-NFE + control -> optional PDD 4-NFE. Independently test
kitchen Triton, then a separately provisioned cu130/Sage environment if the
baseline profile justifies it. Compilation is last, after shapes and patches
stabilize. Restore the untouched original model/sampler path for quality runs.

## Reproducible read-only checks

Run from the local workspace; these commands do not start inference:

```powershell
python server_exec.py "git -C /home/star/agentic-world-comfy/ComfyUI rev-parse HEAD"
python server_exec.py "git -C /home/star/agentic-world-comfy/ComfyUI status --short"
python server_exec.py "env CUDA_VISIBLE_DEVICES= /home/star/agentic-world-comfy/.venv/bin/python -m pip show torch triton sageattention diffusers comfy-kitchen"
python server_exec.py "sed -n '355,390p' /home/star/agentic-world-comfy/ComfyUI/comfy_extras/nodes_easycache.py"
python server_exec.py "sed -n '310,340p' /home/star/agentic-world-comfy/ComfyUI/comfy/ldm/minimax/model.py"
python server_exec.py "sed -n '1,50p' /home/star/agentic-world-comfy/ComfyUI/comfy/quant_ops.py"
```

## Primary sources

Read on 2026-10-08. Built-in web search returned no results; Tavily was used
for discovery, followed by direct publisher README/source inspection. HF model
cards were retrieved through hf-mirror.com; canonical provenance URLs follow.
Mutable upstream READMEs are not a pinned reproducibility artifact. Pin their
commits before any implementation. No community speed claim is presented as a
measurement on our server.

- [S1] Installed native EasyCache: https://github.com/Comfy-Org/ComfyUI/blob/95539f56344958339e39b7582a476267d489b0ee/comfy_extras/nodes_easycache.py
- [S2] Installed H3 model/audio wrappers/PDD heads/attention: https://github.com/Comfy-Org/ComfyUI/blob/95539f56344958339e39b7582a476267d489b0ee/comfy/ldm/minimax/model.py
- [S3] Alibaba PAI PDD publisher: https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/blob/main/README.md
- [S4] PDD community implementation and recipes: https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc
- [S5] ModelTC Turbo publisher and variant table: https://github.com/ModelTC/Minimax-H3-Turbo
- [S6] Separate Larryvrh Turbo implementation: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo
- [S7] MiniMax official architecture and deployment: https://github.com/MiniMax-AI/MiniMax-H3
- [S8] Comfy model publisher quantization guidance: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/README.md
- [S9] Installed kitchen backend gate: https://github.com/Comfy-Org/ComfyUI/blob/95539f56344958339e39b7582a476267d489b0ee/comfy/quant_ops.py
- [S10] SageAttention primary project: https://github.com/thu-ml/SageAttention
- [S11] Installed compile node: https://github.com/Comfy-Org/ComfyUI/blob/95539f56344958339e39b7582a476267d489b0ee/comfy_extras/nodes_torch_compile.py

Not established by these sources: a universal fastest configuration, local
EasyCache speed/quality, local PDD/Turbo + Union 2.0 compatibility, benefit of
four-GPU serving, or a lossless speedup for complex combat actions. Those need
the controlled inference experiments above, assigned to the implementation task.
