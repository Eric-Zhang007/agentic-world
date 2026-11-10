# Agentic World Model Prototype

## October 12 appearance memory

The evidence rollout path retrieves object appearance crops for agent planning,
prompt reasoning and review, then indexes new references only after accepting
and reconciling a video. Renderer image selection is separate: the default H3
input is the accepted full-frame boundary, not every retrieved memory crop.
See `docs/appearance_memory_20261012.md` and
`docs/reference_ablation_20261012.md` for commands, regression evidence and limits.

## October 9 motion-control ablation

The articulated Godot world now exports a clean depth pass for an isolated H3
experiment: dense, key-event and endpoint-only temporal control, two seeds.
The experiment stores outputs and reviews without committing generated world
facts. See `docs/motion_ablation_20261009.md` for implementation, reproducible
commands, input corrections and limitations. Temporal gating is experimental;
it is not a claim of reliable sparse control or endpoint preservation.

## October 9 persistent whitebox

An independent agent-authored Godot whitebox path now constructs parameterized
terrain, scene structures, articulated actors, explicit weapon motion and active
NPCs. Numerical checks and actual multi-view visual review precede each commit.
Two consecutive actions in one persistent world were tested. See
`docs/whitebox_20261009.md` for commands, outputs, failures and limitations.
The initial whitebox milestone was independent of H3; the separate experiment
above now tests its rendered conditions against H3.

## October 8 control and action experiments

Measured results, videos, failures and verification commands:
`docs/validation_20261008.md`.

World-derived Layout, coarse depth, proxy Canny and gray controls now run through
the isolated native Comfy backend. Multi-control strength/sampling windows,
EasyCache, Triton kernels and first-frame guide are explicit experiment settings;
the original quality recipe and environment remain unchanged.
See `docs/world_control_and_action_design.md` and
`docs/h3_acceleration_research.md`. These are not claims of reliable combat or
superiority to CWM. Sparse geometry can worsen generation.

```bash
cd /home/star/agentic_world
.venv/bin/python scripts/run_action_episode.py \
  --image input_data/wukong/tiger_vanguard_source.jpg \
  --goal 'Sidestep, regain balance, then make one staff counterattack and recover.' \
  --output outputs/my_action --max-rollouts 3 --steps 20 \
  --width 640 --height 512 --gpu 2 --perception-device cuda:2 \
  --kernel-backend triton

# Continue an unfinished action; do not silently replace it with a new goal.
.venv/bin/python scripts/run_action_episode.py --resume \
  --output outputs/my_action --max-rollouts 2 --gpu 2 --perception-device cuda:2 \
  --kernel-backend triton
```

The new action runner creates its own world/observation SQLite, image-derived
INFERRED geometry, original-image DINO features, action phases and completion
criteria. Only accepted completed intended transitions advance the world;
generated-only extras never spawn facts. `action_state.json` and
`run_summary.json` distinguish COMPLETE from budget-exhausted RUNNING.
Each segment preserves candidate timeline, controls, rendered video, masks,
embeddings, matching, discrepancy, raw/effective decision and before/after world.
`--cancel` explicitly cancels; subsequent `--goal` retains the same world.
Fine body/weapon animation remains renderer freedom, not proxy skeletons.

For quality restore `--kernel-backend default --cache-threshold 0 --steps 40`.
Triton is a tested alternative backend, not a dependency/environment upgrade.
EasyCache is opt-in; do not combine it with distillation adapters blindly.

This prototype keeps authoritative world facts separate from generated-video
observations. The supplied GTA project is the source of the canonical entity
contract; `video_to_whitebox` scenes are imported as `INFERRED` approximate
geometry with their uncertainty assumptions preserved.

## Server

The implementation is deployed under `/home/star/agentic_world`. Existing H3
is read-only at `/home/star/minimax-h3`; the selected renderer is the existing
INT8 full Ref2VA recipe on physical GPUs 2/3. H3 accepts prompt plus image/video
references, not depth or semantic maps in this recipe. The compiler emits proxy
RGB, semantic and depth artifacts for inspection, but the wrapper refuses to
pretend unsupported maps were conditioned into H3.

An isolated native ComfyUI backend now exists at
`/home/star/agentic-world-comfy`, with `renderer/comfy_h3.py` as its Python wrapper.
It pairs the already-present pruned INT8 Ref2VA with Union 2.0 pruned INT8
ControlNet. Real Canny-plus-reference generation is verified; see
`docs/comfy_control_trial.md` for commands, artifacts and scope. This is not the
default prototype renderer, and source-derived Canny does not establish
world-derived depth/semantic conditioning.

Recorded-world projection/time calibration and HUD exclusion are implemented
in `compiler/episode_alignment.py` and `perception/hud.py`. The current calibration
passes held-out center-projection checks only for agent_001 in the inspected
episode prefix; agent_000 is rejected for insufficient evidence. See
`docs/calibration_and_hud.md` for parameters, limits, overlays and the real
40-step HUD-cleaned rollout. The default CLI does not silently select this
recording-specific calibration or HUD mask for unrelated inputs.

```bash
cd /home/star/agentic_world
/home/star/agentic_world/.venv/bin/python scripts/run_prototype.py \
  --config configs/default.yaml --steps 50 --max-reroll 1
```

Three-step H3 smoke completes in minutes; this image-reference 50-step rollout
measured about 217 seconds per renderer subprocess. The old video+audio reference
benchmark takes about 12 minutes and is not directly comparable.
The first successful smoke artifact is under `/home/star/agentic_world/outputs/h3_smoke`.

## Architecture

`WorldDB` is authoritative and records edits. `ObservationStore` contains only
sampled generated frames, detected objects and reports. Matching uses Hungarian
assignment with category, appearance, spatial, projection and temporal terms
when available. Perception uses real GroundingDINO tiny, SAM2.1 tiny and DINOv2
small sequentially; missing weights fail rather than producing fake observations.
Models reuse `/mnt/hdd/agentic-world-models`; downloads use
`HF_ENDPOINT=https://hf-mirror.com`. SAM3 and DINOv3 require gated access.
A generated-only unmatched object is never inserted into `WorldDB`.
The Agent sees a compact summary, relevant RGB frames, discrepancy and query tools,
not a serialized world in conversational history, and returns
`ACCEPT`, `RECONCILE_WORLD`, or `REGENERATE`. Regeneration is bounded by three
rerolls (four total attempts). The command above limits this to one reroll.
Seeds change across rerolls. Exhaustion remains REGENERATE, not ACCEPT.

## Verification

```bash
cd /home/star/agentic_world
.venv/bin/python scripts/test_cases.py
.venv/bin/python scripts/test_cases.py --agent
.venv/bin/python scripts/test_h3.py --output outputs/new_smoke --steps 3
.venv/bin/python scripts/run_prototype.py --output outputs/replay_new \
  --replay-video outputs/h3_smoke/generated.mp4 --max-reroll 0
.venv/bin/python scripts/inspect_world.py data/world.sqlite
.venv/bin/python scripts/probe_perception.py --image /path/frame.png \
  --output outputs/perception_check
```

Use a new output directory each time. Persistent world objects survive subsequent
runs; the demo scene is initialized only into an empty database.
`agent/world_agent.py` provides query/spawn/update/move/remove/set_attribute,
render/accept/reroll tools. Mutations require object-specific independent
original/manual evidence matching the exact change. The current GPT loop exposes
read-only tools and executes structured, evidence-backed update proposals afterwards.
Configuration contains the user-requested gateway/model. `OPENAI_API_KEY` can
override the configured credential; do not publish the credential-bearing config.

## Data and schema

See `docs/world_state_schema.md`, `docs/gta_database_inspection.json` and
`docs/input_data_assessment.md`. The supplied SQLite stores episode lifecycle,
deduplication and clock metadata, while canonical entity payloads are JSONL/
Parquet derived artifacts. `inspect_world.py` prints tables/schema/count/sample.

The inverse scene renderer is reused from the supplied package:

```bash
.venv/bin/python scripts/test_inverse_compiler.py \
  --scene outputs/inverse_source_scene.json --source vendor/video_to_whitebox \
  --output outputs/inverse_check_new
.venv/bin/python scripts/run_prototype.py \
  --inverse-scene outputs/inverse_source_scene.json \
  --whitebox-source vendor/video_to_whitebox --timestamp 1 \
  --world-db data/inverse_new.sqlite --output outputs/inverse_episode_new \
  --max-reroll 0
```

The inverse integration freezes one supplied timestamp into a working state;
it does not claim accurate reconstructed dynamics. Unknown semantic labels stay
unknown. Use a new working DB for import so existing world state is not replaced.
See `docs/validation_report.md` for measured results and remaining test gaps.
Follow-up diagnostics, original-RGB checks and video-reference smoke are in
`docs/followup_validation.md`. Reports now distinguish conflicting detector
labels and objects already undetectable on the compiled reference.

Silent reference-video baseline:

```bash
.venv/bin/python scripts/test_video_reference.py outputs/gta_sampled_reference.mp4 \
  --output outputs/video_reference_new --steps 3
.venv/bin/python scripts/test_perception_quality.py
.venv/bin/python scripts/test_reference_sampling.py
```

This uses the installed Ref2VA video input, not ControlNet. Three-step results
only validate execution. Do not interpret resampled 5fps RGB as native 24fps data.

## Output

For the separate video-acceptance loop, see
[`docs/video_accept_reconcile.md`](docs/video_accept_reconcile.md). Its entry
point is `scripts/run_video_reconcile_trial.py`, which requires a fresh output
directory. It generates a pending plan, renders H3, asks the configured GPT to
review sampled video, commits corrected actor trajectories/events/end state,
and only then plans and renders a continuation from that committed state.
Partial acceptance advances the attained state. Rejection leaves it unchanged.
These corrections are labeled agent estimates, not measured 3D reconstruction.
The older experiment entry points keep their original behavior.
Persistent standing/crouched/low-guard posture, support templates, and the
separate continuation experiment are documented in
[`docs/posture_reconcile.md`](docs/posture_reconcile.md).
Coarse target-arrival checks and the independent optional temporal-inspection
reviewer are documented in [`docs/arrival_acceptance.md`](docs/arrival_acceptance.md).
The preceding evidence pilot and outstanding interaction acceptance cases are in
[`docs/evidence_skills_acceptance.md`](docs/evidence_skills_acceptance.md).
The actual server result, including the accepted first segment and rejected
continuation, is recorded in
[`docs/video_accept_reconcile_validation_20261009.md`](docs/video_accept_reconcile_validation_20261009.md).

Each episode contains `attempt_XX/proxy`, `generated.mp4`, sampled frames,
`observations.json`, `discrepancy.json` and `agent_decision.json`, plus
`world_before.json`, `world_after.json` and `agent_decisions.json`.
`matching.json` holds per-frame assignments; `segmentation/<frame>/` holds masks,
crops, embeddings and logs. `sampled_frames/debug_*.png` compares solid observed
boxes with dashed expected boxes. Generation request/timing/exit status are saved.

## Limitations

Continuous root-motion boundaries, the prompt/whitebox generation skill, and
native historical video prefixes are documented in
[continuous motion](docs/continuous_motion.md). Real visual acceptance must be
checked separately from its state and time bookkeeping tests.

For the evidence-driven static geometry repair and historical acceptance workflow,
see [scene refinement](docs/scene_refinement.md). It distinguishes revisions of
inferred geometry from events in world time and retains the original history.
The isolated server proxy setup is documented in [server proxy](docs/server_proxy.md).

The original DiffSynth recipe does not accept explicit depth/semantic conditions;
its first-frame input is a Ref2VA image reference, not locked frame zero.
The separate Comfy path supports the tested Union RGB controls and a native
frame-zero guide; this does not imply exact pixel locking or semantic-ID support.
Proxy depth/semantic maps are box approximations, not accurate surface renders.
The configured open-vocabulary proposal list can miss unknown classes outside it.
SAM improves masks, not incorrect detector labels. Depth stays unset without
metric calibration. DINOv2 proxy-to-generated matching has domain shift.
Tracking is consecutive sampled-frame assignment, not dense tracking.
The legacy generic entity-edit tools require independent evidence; generated
repetition alone is insufficient. The separate whitebox video-acceptance path
reconciles supported actor state from explicitly accepted video, preserves old
history, and labels uncertain single-view estimates.
The inverse
whitebox data is useful for scaffolding and consistency checks, not reliable
ground truth.
