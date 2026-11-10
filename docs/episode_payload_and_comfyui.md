# Episode payload and ComfyUI investigation

Inspected on 2026-10-03. No ControlNet weights were downloaded and no H3
environment or renderer was modified during this investigation.

Subsequent update: a separate user-authorized ComfyUI trial DID download the
missing control branch and complete real generation. See `comfy_control_trial.md`;
the no-download/no-inference statements below describe this earlier inspection only.

## The new archive contains real world payloads

Input: `C:\Users\zjc\Downloads\ep_20260812_090725_f8a5.zip` (2,127,451,365 bytes).
This supersedes the missing-payload conclusion for the earlier two archives;
it does not change what those earlier archives contained.

- `.spool/world_state.jsonl`: 46,318,974 bytes, 17,540 parsed snapshots.
- `server/world_state.parquet`: 17,540 rows with actual `payload_json`.
- `server/entities.parquet`: seven lifecycle records.
- World sample sequences are contiguous, 0 through 17,539.
- First-to-last world timestamp span: 1753.8657709 seconds.
  The manifest reports 1753.9657709 seconds including one sample interval.
- Two agent streams, each with camera/actions/state/frame timestamp Parquet.
- 436 MKV segments for agent_000, 401 for agent_001.

World wire schema is 0.1.0. Each snapshot includes sample sequence, server
monotonic/wall timestamps, environment, agents/vehicles/peds/objects, task state,
and relation_edges. Entity payloads contain dataset_entity_id, network_id,
model_hash, position, native Euler rotation, velocity and observable state.
Unavailable fields are explicitly null or marked unavailable.

Whole-episode entity IDs: six in agents, one in vehicles, none in peds/objects.
At most two agents and one vehicle occur in a single sample. Six agent entity
IDs do not mean six simultaneous participants or necessarily identity corruption.
Keep episode-scoped dataset IDs and lifecycle metadata.

All snapshots declare `world_state_kind=observable` and completeness `partial`.
This is a useful recorded dynamic state, NOT a complete static map/mesh or an
omniscient simulation dump. Trees/buildings/props absent from these lists are
not thereby absent from the world. Such detections remain unmatched observations;
absence alone is insufficient to classify them as renderer hallucinations.

## Quality and calibration limits

The archive's own quality report says `valid=false`:

- Recording covers about 29 minutes of a planned two hours.
- Reported drop rate: 0.8381%; action missing rate: 5.7241%.
- Reported cross-agent alignment: 42.1674 ms.
- Reported agent-world presence: 96.0490%.
- No reported world-snapshot gaps, ID collisions or identity fragmentation.

These quality metrics are recorder assertions, not independently remeasured
across all video segments in this investigation.

Camera payloads contain position, Euler angles, FOV, render_resolution and
frame_counter, but no supplied calibrated K/view matrices. Initial FOV is 50.
Initial render sizes are 1920x1080 and 2560x1600; decoded recordings are both
1280x720. Determine resize/crop behavior, FOV axis and Euler/extrinsic convention
before asserting quantitative projection consistency.

Aligned timestamp rows contain estimated server time and clock alpha/beta,
but no explicit segment-local decoded-frame identity in the inspected row.
First timestamp row is frame_index=19 and video_pts about 0.96 seconds, while
the first decoded MKV frame is at 0.1 seconds. Do not equate world row zero,
timestamp row zero and decoded frame zero. Join via calibrated clocks/PTS and
verify segment offsets and continuity before generating controls.

## Server verification

Subset: `/home/star/agentic_world/input_data/ep_20260812_090725_f8a5`.
Local evidence: `input_data/ep_20260812_090725_f8a5/{inspection,probe}.json`.
Extracted only Parquet, small metadata and the first MKV from each agent.
The complete ZIP and JSONL remain on the local machine; Parquet contains the
complete finalized world snapshots.

`probe_episode_subset.py` read all selected Parquet tables, converted the first
world payload using the original GTA adapter, and fully decoded both selected
videos. Each is 80 frames at 20 fps, 1280x720, PTS 0.10..4.05, strictly increasing.
The adapter returned two episode-scoped `person` objects with OBSERVED provenance
and original world positions. This is an adapter/media check, not a new H3 rollout.

Added pyarrow 25.0.1 and pydantic 2.13.5 to the isolated prototype venv.
Pydantic dependencies: annotated-types 0.8.0, pydantic-core 2.46.5,
typing-extensions 4.16.0 and typing-inspection 0.4.4. H3 venv was not changed.

```bash
python scripts/inspect_episode_zip.py EPISODE.zip --output EXTRACTED_SUBSET
python scripts/probe_episode_subset.py EXTRACTED_SUBSET --source-root GTA_SOURCE
```

## ComfyUI materially changes the implementation assessment

Primary-source inspection: Comfy-Org/ComfyUI PR #16471,
merged 2026-09-22T16:27:10Z, commit
`95539f56344958339e39b7582a476267d489b0ee`.
Files were read at this pinned commit, not inferred from third-party tutorials.

- `comfy_extras/nodes_minimax_h3.py:579`: native
  `MiniMaxH3FunControlNetApply` accepts model, model_patch, VAE, strength,
  start/end percent, optional control_video, mask and source_video.
- `comfy_extras/nodes_model_patch.py:289`: detects control block count and
  assigns five blocks every ten base blocks for v1, ten every five for v2.
- `comfy/ldm/minimax/controlnet.py`, `init_stream`: uses layout.img_update
  to assign target control rows, leaving reference/keyframe control rows zero.
  Thus this implementation explicitly handles the row layout that the installed
  DiffSynth path did not handle. This is source-level evidence, not GPU validation.
- The same method checks timestep/adaLN dimensions and fails for full-vs-curve
  mismatches. Changing frontend does not make arbitrary checkpoints compatible.
- The model-patch loader detects quantized layers and metadata for adaLN basis,
  control block placement and post-normalization inpaint masks.

The author's Union 2.0 model card documents Depth, Canny, HED, MLSD, Pose,
Scribble, Layout and Gray. Layout is color-coded boxes on white RGB, NOT integer
semantic-ID/instance-ID conditioning. Raw camera matrices or metric depth arrays
are not direct inputs to this Comfy node either; the compiler is still needed.

Kijai's experimental repository lists v2 pruned BF16 and pruned INT8 convrot
model patches. Their existence does not prove they match the current unpruned
INT8 base. Inspect tensor shapes and metadata before selecting or converting.
The original v2 branch is about 13.55 GB; only metadata/model cards were fetched.

Recommended next experiment: an isolated ComfyUI backend pinned to verified
code, reusing existing base weights where compatible, with one short Depth or
Layout control and a no-control baseline at the same seed. Validate both token
layout and quantization compatibility before a full 50-step comparison.
Keep the existing DiffSynth renderer as a working fallback. No ComfyUI service,
model download or control-conditioned GPU inference was performed here.

Primary sources (URLs preserved for reproducibility):

```text
https://github.com/Comfy-Org/ComfyUI/pull/16471
https://github.com/Comfy-Org/ComfyUI/tree/95539f56344958339e39b7582a476267d489b0ee
https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0
https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/model_patches
```

Built-in web lookup returned no usable content; Tavily located sources and
GitHub API/raw source plus HF mirror model cards were then inspected directly.
