# October 8, 2026: measured controls and persistent actions

## Scope and environment

Deployed at `/home/star/agentic_world`; original `/home/star/minimax-h3`
was not modified. Native Comfy runs in `/home/star/agentic-world-comfy`
at commit `95539f56344958339e39b7582a476267d489b0ee` (clean worktree).
Existing INT8 Ref2VA/Union, GroundingDINO, SAM2.1 and DINOv2 caches were reused.
No packages or model weights were installed/downloaded in this iteration.
Triton uses the existing backend option, not a torch/CUDA upgrade.
All four GPUs returned to 15 MiB / 0% utilization after testing.

## Controls: actual execution versus useful control

Final experiment: `outputs/world_controls_outline_20261008/render`,
832x480, 73 frames, 24 fps, 20 steps, seed 43, Triton and first-frame guide.
Controls are derived from calibrated GTA objects and camera data, not source
HUD pixels. Three generated frames per case underwent actual perception/matching.
The expected world is partial; errors below concern its visible expected actor.

| Condition | Sampling seconds | Sampling + decode/save seconds | Missing ratio | Matched center error px |
| --- | ---: | ---: | ---: | ---: |
| Reference only | 50.61 | 61.60 | 0 | 31.03 |
| Layout | 59.63 | 66.21 | .333 | 37.63 |
| Depth | 58.76 | 65.60 | 0 | 102.11 |
| Proxy Canny | 58.87 | 65.40 | .333 | 146.23 |
| Gray | 58.84 | 65.15 | 0 | 67.84 |
| Layout + depth | 74.16 | 80.66 | .333 | 63.68 |
| Layout + EasyCache | 34.25 | 40.89 | 0 | 102.94 |

EasyCache skipped 10/20 evaluations: 1.74x sampling acceleration for the final
Layout pair, but spatial error worsened. These are single-seed short clips,
not total pipeline times or a general quality ranking. Missing matches bias
conditional center-error means. Reference-only was best on this partial scene.
Actual ControlNet calls/residuals were recorded; execution is verified, uniformly
better adherence is not. Raw semantic-ID and skeleton controls are not claimed.

An earlier filled-box Layout implementation was wrong for the official VACE
outline-on-white format. It was fixed and rerun; old experiments remain under
`outputs/world_controls_20261008` as negative controls. On identical old inputs,
Triton reduced sampling 131.61 -> 65.96 seconds; Triton + cache reached 37.90
seconds (3.47x versus eager). This is a backend benchmark, not evidence that
the old filled Layout was correct. The GTA source itself also contains a
ghosted/semitransparent player, so that artifact cannot all be assigned to H3.

Recommended debugging setting: Triton, cache off initially, coarse selective
controls. Restore the original quality recipe with 40 steps, default backend,
cache threshold 0. See `h3_acceleration_research.md` for primary sources;
PDD/Turbo/Sage alternatives were researched, not deployed or benchmarked.

## Persistent action test

Input: public real gameplay screenshot, recorded in
`input_data/wukong/source.json`. All initial camera/depth geometry is INFERRED.
Run: `outputs/action_wukong_outline_20261008`.

First goal was split by GPT into left dodge (73 frames), regain balance
(73 frames), and one counterattack with recovery (124 frames): 11.25 seconds.
All three phases passed the coarse visual critic and committed in order.
The counterattack is small/foreshortened and late; this is not verified contact,
damage or victory. Nine review frames still leave action ambiguity.

A second goal, measured backward step with guard and a modest camera retreat,
continued the same world and IDs. Two attempts did not commit. A critic bug was
found: it compared continuity to the original photo rather than the latest
committed image. After separating identity reference from continuity reference,
the third attempt completed. Its accepted 73-frame clip is
`action_001_segment_002/render/controlled.mp4`.

World time advanced 0 -> 3.0417 -> 6.0833 -> 11.25 -> 14.2917 seconds.
There were zero inserted or removed IDs. The last transition moved both fighter
and held staff; held props now inherit holder translation. Shared approximate
initial depth for held props was also fixed/tested, but the first episode was
initialized before that fix. No retroactive geometry correction is claimed.

`review.json` records all six attempts and their diffs. `accepted_action.mp4`
concatenates only four accepted completed phases across the two goals, not
rejected trials. Local `first_action_11s.mp4` preserves just the first goal;
local `retreat.mp4` is the second accepted action.

This proves orchestration and persistent-state behavior, not reliable spatial
verification: the final retreat still had missing ratio .667 and mean matched
projection error 106.36 px, with conflicting detector labels. GPT accepted coarse
visible motion despite those unresolved identity/geometry discrepancies.

GTA movement trials were rejected; the separate 40-step glance/nod trial
completed only its first phase. `outputs/action_gta_glance_20261008` remains
RUNNING with user_input_ready=false. It is not a completed GTA action claim.
Older Wukong strict-grip trials likewise remained incomplete. Those failures
motivated coarser outcome criteria rather than invented exact joint poses.

## Implementation and verification

Added `compiler/world_controls.py`, `agent/action_planner.py`,
`state/action_session.py`, and scripts `prepare_world_control_trial.py`,
`evaluate_world_controls.py`, `summarize_control_trials.py`,
`run_action_episode.py`, `review_action_episode.py`, `test_action_controls.py`.
Modified `renderer/comfy_h3.py`, `scripts/run_comfy_control.py`, and
`state/db.py` for controls, measured acceleration and atomic phase commits.
Added two experiment configs and design/acceleration documentation. The local
SSH helper now sends keepalives. No mature external agent harness was installed.

Server verification: 51 tests passed; compileall passed.

```bash
cd /home/star/agentic_world
.venv/bin/python -m unittest scripts.test_action_controls scripts.test_calibration_hud scripts.test_cases scripts.test_perception_quality scripts.test_reference_sampling scripts.test_comfy_renderer
.venv/bin/python -m compileall -q agent compiler renderer scripts state
.venv/bin/python scripts/review_action_episode.py outputs/action_wukong_outline_20261008
```

Tests cover rejected/partial transitions, transaction rollback, phase gates,
held-object propagation, outline format and critic reference separation.
README contains fresh-run and resume commands. Each segment retains candidate
world/timeline, controls, video, perception, matching, discrepancy and decisions.

## Next priorities

1. Calibrate identity/visibility and camera-relative geometry; audit critic
   acceptance against matching failures and denser action evidence.
2. Preserve verified partial progress and implement overlap-based continuation;
   currently incomplete phases retry conservatively from committed state.
3. Build same-input/action regression comparisons before choosing a Pi/Codex
   runtime or claiming improvements over CWM. Its trained semantic/depth LoRA
   is not equivalent to this generic Union conditioning.
