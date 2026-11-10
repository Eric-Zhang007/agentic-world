# Geometry Proposal Tool

Entry point: `whitebox.geometry_pipeline.propose(args: Mapping) -> dict`.
This is a synchronous, local-host callable for the Pi `geometry_propose`
operation. Run it in a worker when the caller has an async event loop.
It does not register tools, write a database, accept proposals, merge worlds,
execute paid APIs, or change existing compiler/renderer code.

## Calling Contract

```python
from whitebox.geometry_pipeline import propose

result = propose({
    "image": "/absolute/path/input.png",
    # Omit either cache to run the corresponding existing inference script.
    "reconstruction": "/absolute/path/reconstruction.npz",
    "perception": "/absolute/path/perception/result.json",
    "frame": 0,
    "output": "/absolute/path/NEW_proposal_directory",
})
```

All paths refer to the host executing `propose`, not the Pi client's host.
Relative input/output paths resolve against the caller's current directory.
Subprocesses run from the repository root. Output must not already exist;
omitting output allocates `outputs/geometry_proposals/<uuid>/`.
The module never deletes or reuses an existing output directory.

| Argument | Default / meaning |
|---|---|
| `image` | Required RGB image path, including for cached runs |
| `reconstruction` | Optional existing VGGT NPZ; otherwise reconstruct this single RGB |
| `perception` | Optional existing perception result JSON; sibling `input.png` and masks must remain available |
| `frame` | 0; selected cached reconstruction frame, nonnegative integer |
| `output` | New unique directory if omitted |
| `python` | `sys.executable`; compiler and renderer interpreter |
| `reconstruction_python`, `perception_python` | Default to `python`; separate environments supported |
| `gpu` | `"2"`; single physical GPU index; all child processes get `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<gpu>` |
| `device` | `"cuda:0"`; logical perception device mapped to the selected physical GPU; also accepts `cuda` or `cpu` |
| `text` | `person. car. building. road. sidewalk. tree. pole. wall.` |
| `threshold` | 0.25 detector threshold |
| `scale`, `scale_source` | Compiler defaults: 1, uncalibrated model units |
| `actor_height_prior` | Omitted; only explicitly supplied positive prior is used |
| `confidence_quantile` | Compiler default 0.2 |
| `plane_threshold` | Compiler default 0.02, in model units |
| `preview_scale` | 4; integer 1-4 times source intrinsic dimensions, same aspect ratio; renderer limit 4096 |
| `timeout` | 600 seconds per subprocess, not whole-request deadline |
| `godot` | Optional executable forwarded to existing renderer |
| `use_display` | False; true reuses an existing X display (including local WSLg) instead of starting Xvfb |

Unknown options fail validation. Nonzero frame without a cached reconstruction
is rejected. There is deliberately no `accept`, `database`, shell command, HUD
inpainting, automatic road retry, or arbitrary extra-arguments parameter.

```json
{
  "success": true,
  "status": "PROPOSAL_READY",
  "accepted": false,
  "world_merge_ready": false,
  "database_written": false,
  "scene_path": "/run/candidate/scene.json",
  "review_images": ["/run/preview/camera.png", "/run/preview/overview.png"],
  "preview": {
    "manifest": "/run/preview/manifest.json",
    "images": ["/run/preview/overview.png", "/run/preview/front.png",
               "/run/preview/side.png", "/run/preview/camera.png"]
  },
  "evidence": {
    "pipeline_report": "/run/proposal.json",
    "reconstruction": "/cache/reconstruction.npz",
    "perception": "/cache/perception/result.json",
    "compiler_report": "/run/candidate/report.json",
    "source_candidates": "/run/candidate/source_candidates.json"
  },
  "stages": {},
  "timing_seconds": {},
  "warnings": []
}
```

The example is abbreviated: evidence also contains RGB correspondence status,
detected categories, ground inference, facade coverage, entity count and scale
source. Stage entries contain `status`, `seconds`, and, for subprocesses,
`command`, `log`, and `returncode` when available. Cached stages have zero
execution seconds and `status="cached"`; these are NOT measured model timings.
All results are JSON-safe and persisted in `proposal.json` after output creation.
On success `review_images` always contains exactly two existing absolute image
paths in camera/overview order, for the existing Pi image-content channel. The
other views remain available under `preview.images` without being attached.
On failure `review_images=[]`; a manifest missing either required review view
is a preview-stage failure, even if other images exist.

Operational failures return `success=false`, `status="FAILED"`, and
`error={stage,type,message}`. Validation failures before output creation have no
report file. Compiler failures return no scene; renderer failures can retain
`scene_path` but return `preview=null`. A caller must check `success`, not just
whether a scene file exists. Partial artifacts and logs remain for diagnosis;
no stale previews are presented as successful results.

## Existing Implementations and Evidence Boundaries

1. `scripts/reconstruct_vggt.py`: single RGB, `--local-files-only`.
2. `scripts/probe_perception.py`: GroundingDINO/SAM2/DINOv2, offline local models.
3. `scripts/compile_geometry_assets.py`: ground alignment, deduplication, local
   facades, original unassigned candidates, inferred ground evidence.
4. `python -m whitebox.godot.render`: unchanged scene, four review views.

The caller RGB must equal perception `input.png` pixel-for-pixel after RGB
conversion, even when both caches agree with each other. The existing compiler
then verifies the selected reconstruction RGB against perception RGB after its
recorded resize/crop/padding and checks original image dimensions. It also
transforms masks with the existing pixel-center mapping. Filenames alone never
establish correspondence. Cached NPZ frame bounds and malformed arrays remain
the compiler's responsibility.

Missing exact `road`, `ground`, or `sidewalk` categories fails explicitly. A
mixed detector label is not silently relabeled. Detected ground can still fail
confidence filtering or plane fitting; the compiler's error is preserved.
The old separate road pass was a manual experiment, not a general algorithm.
The tested perception cache includes that historical road pass, so these tests
do NOT establish that fresh automatic segmentation reliably finds roads.

All model subprocesses receive offline environment flags. GPU visibility is
set in child environments only; the parent's environment is not changed. Use
`gpu="2", device="cuda:0"`, NOT `device="cuda:2"`. GPU mapping and offline flags
are recorded per stage. The current
perception script requires `/mnt/hdd/agentic-world-models/models.json`; models,
GPU, Python dependencies and Godot/Xvfb must already be installed. Missing
dependencies are reported, not installed. The module does not transport inputs
or deploy a project. Call it on the server with server-local paths.

Scale is not known from one RGB. Facade thresholds in the existing compiler can
produce poor coverage in uncalibrated units. Explicit height/scale priors must
remain labeled as assumptions. Low coverage is returned as a warning and the
original building proposals remain in `source_candidates.json`; unfitted points
are unknown, not deleted buildings or accepted collision geometry.

The source dataset proxy and interpolated recorded camera are NOT reliable
visual GT. RGB pairing, successful rendering and internal projection checks
are not independent geometric accuracy, visual alignment or world acceptance.
Ground inference remains explicitly inferred. No hidden thickness is measured.

## Actual Verification: 2026-11-04

Local and server: `python -B scripts/test_geometry_pipeline.py`, **10 tests pass**.
Coverage includes fresh/cached orchestration, offline flags, separate Python
executables, caller RGB mismatch, missing ground, compiler failures, preview
timeout/missing images, missing compiler output, cached frame forwarding,
invalid options and refusal to overwrite existing work. Unit-test model subprocesses
are mocked. These initial tests were followed by real fresh inference below.

Real server runs used existing NPZ/perception caches and the actual compiler and
Godot. Paths below are relative to `/home/star/agentic_world/`:

| Run | NPZ | Compile | Preview | Total | Result |
|---|---|---:|---:|---:|---|
| `outputs/geometry_pipeline_tests/server_single_final` | `outputs/vggt_20261103/urban_single.npz` | 1.668s | 3.220s | 4.960s | Four views, 68 entities / 58 facade parts |
| `outputs/geometry_pipeline_tests/server_joint3_final` | `outputs/vggt_20261103/joint3.npz`, frame 0 | 1.668s | 4.224s | 5.962s | Four views |
| `outputs/geometry_pipeline_tests/server_wrong_frame` | joint3, deliberately wrong frame 1 | 0.416s | Not run | 0.516s | Compiler correctly rejected RGB mismatch |

Both successful runs explicitly used `plane_threshold=0.005` and
`actor_height_prior=1.7`; these are NOT new pipeline defaults. Single-frame
facade point coverage ranges from 0.120 to 1.0. There are 102,052 ground
triangles and 1,827 inferred ground pixels. These figures describe the candidate,
not scene completeness. Source-camera preview was visually inspected; substantial
unknown background and coarse geometry remain visible.

Earlier 518x294 and 1036x588 four-view renders failed the existing renderer's
blank/near-uniform check on distant overview/front views. At 2072x1176 the same
unaltered single-frame candidate passes: changed samples are 287/108/839/21320
for overview/front/side/camera, respectively. This motivates the explicit default
`preview_scale=4`; no detector threshold or renderer code was changed. Render
success does not imply good framing for every future scene. Failed trial outputs
are preserved. A trial preview-scene reframing helper was removed, not shipped.

Local evidence copies: `outputs/geometry_pipeline_tests/server_single_final_report.json`,
`server_joint3_final_report.json`, `server_single_manifest.json`, and
`server_single_camera.png`. The server run directories contain full candidate,
logs, source candidates and preview files. A local real compiler run also passed;
local preview correctly failed because Godot is unavailable on that machine.

Reproduce the server cached single-frame test (choose a NEW output each time):

```bash
.venv/bin/python -B scripts/test_geometry_pipeline.py --real \
  --image outputs/geometry_init/urban/perception_target/input.png \
  --reconstruction outputs/vggt_20261103/urban_single.npz \
  --perception outputs/geometry_init/urban/perception_target/result.json \
  --output outputs/geometry_pipeline_tests/new_single_run \
  --plane-threshold 0.005 --actor-height-prior 1.7
```

Only the new module and its test script were uploaded individually for these
tests; no project-wide synchronization, existing code overwrite, DB changes or
paid API calls were performed. The documentation is a separate new file.

## Fresh GPU2 Verification: 2026-11-04

Both fresh runs omitted `reconstruction` and `perception`. The input was
`/home/star/agentic_world/outputs/gta_pi_loop_20261102/urban/input.png`.
The Pi deployment's default world interpreter,
`/home/star/agentic_world/.venv/bin/python`, successfully ran VGGT, perception,
compiler and renderer. No separate Python environment was necessary, and no
dependencies or weights were installed/downloaded. The venv already reads
shared Torch packages through its existing read-only path; no H3 inference,
configuration or environment modifications were performed.

The first real run exposed a deployment risk in the old orchestration: `gpu`
was forwarded only to VGGT while perception inherited CUDA visibility. This was
fixed before inference by explicitly restricting all children to the selected
physical GPU. `gpu=2` now maps perception `cuda:0` to physical GPU2. The VGGT
metadata records `physical_gpu="2"`; perception records
`cuda_visible_devices="2"` and `device="cuda:0"`. No process was assigned to
GPU0 by this task; the main thread's GPU0 experiment was left alone.

VGGT used its existing local snapshot:
`/home/star/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9`.
Perception loaded the detector/SAM2/DINOv2 paths already registered in
`/mnt/hdd/agentic-world-models/models.json`. Offline flags and VGGT
`--local-files-only` were enabled throughout. The VGGT log's generic message
mentioning a download endpoint is not a download: it is immediately followed by
local-directory weight loading.

Final invocation parameters are stored in
`outputs/geometry_pipeline_tests/fresh_gpu2_final_args.json`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -B -m whitebox.geometry_pipeline \
  --args outputs/geometry_pipeline_tests/fresh_gpu2_final_args.json
```

The explicit geometry parameters were `plane_threshold=0.005` and
`actor_height_prior=1.7`; no detector thresholds, prompt defaults, or ground
fallbacks were changed. Use a new output path when repeating the command.

| Stage | First fresh run | Final fresh run |
|---|---:|---:|
| VGGT subprocess | 14.496s | 14.499s |
| Perception subprocess | 9.986s | 8.985s |
| RGB/mask check | 0.158s | 0.141s |
| Compiler | 1.567s | 1.517s |
| Preview | 3.171s | 2.670s |
| Total | 29.384s | 27.818s |

The final VGGT model forward/camera decode alone took 0.574s; this is NOT the
14.499s whole subprocess cost. Final detector/SAM/embedding reported
1.770/0.369/0.759s internally, excluding remaining process/loading overhead.
Both runs completed without subprocess errors. The final run also verifies the
new explicit ground warnings. Local and server regression suites now contain
13 passing tests, including GPU isolation without parent-environment mutation,
mixed-ground labels not promoted to ground, and sidewalk-only warning behavior.

### Execution Success Is Not Geometry Acceptance

The fresh default prompt produced 21 observations, including `road sidewalk`
and a separate `sidewalk`, but **no exact `road` category**. Only the sidewalk
was used as ground. It yielded 4,271 ground triangles, zero inferred pixels and
zero inferred triangles. Both person support requests and the car support
request were rejected with `no_bracketed_ground_support`.

The existing compiler retained `geo_007`, category `road sidewalk`, as a solid
box of approximately `[26.686, 1.136, 6.663]` in the assumed-height scene units.
The source-camera preview visibly shows this thick box occluding the lower
bodies. This is **bad road geometry**, not a repaired road or a complete world.
The pipeline reports `PROPOSAL_READY` only because candidate and four previews
exist; `accepted=false` and `world_merge_ready=false` remain unchanged. It now
explicitly reports `ground_segmentation.exact_road_detected=false`, the mixed
label, and rejected support warnings. No relabeling, manual road pass, hidden
retry or geometry alteration was performed. Compiler semantics remain a
separate owner task; do not accept this result as navigable ground.

Final server output directory:
`/home/star/agentic_world/outputs/geometry_proposals/fresh_gpu2_20261104_02/`.
Main preview: `preview/camera.png`; other views: overview/front/side.
Full evidence: `proposal.json`, `reconstruction.metadata.json`,
`perception/result.json`, `candidate/report.json` and per-stage logs.
Local evidence copies are `outputs/geometry_pipeline_tests/fresh_gpu2_*`.
The earlier `_01` output is retained unchanged. No accepted world or database
was changed.
