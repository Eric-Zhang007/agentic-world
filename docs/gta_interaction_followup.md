# GTA interaction follow-up

本轮新增 4 次真实 H3 生成：乡镇 1 次拒绝、2 次部分接受，城区 1 次接受。
乡镇历史推进到 18 秒，但还没到路边；城区推进到 12 秒，NPC 从冻结跑姿转为移动。
状态接续成立，步态质量没有完全过关：乡镇滑步明显，城区后半段仍僵硬。
城区 agent 判定动作完成，外层视觉复核只认可粗位移和状态关系，不把它算作自然步态成功。

These trials continue the accepted worlds from `gta_pi_loop_20261102` using the
formal Pi harness. The original databases and histories remain unchanged.
Pi writes actions, prompts and reviews; Codex reviews the resulting media and
fixes shared interfaces. No external review API is used by Codex.

## Trials

| Branch | Starting world | New generation budget | Recipe |
|---|---|---:|---|
| `rural/road_edge` | `rural/continuation`, revision 4, time 12s | 3 | 512x288, 20 steps, GPU 2 |
| `urban/npc_walk` | `urban`, revision 3, time 9s | 1 | 1024x576, 40 steps, GPU 0 |

Both use native whitebox video, accepted RGB first frame, depth strength 0.5,
cache disabled and no extra appearance reference images. Actor roots and
prospective NPC behavior are planned; detailed joints are not added.

Rural must continue toward the near road edge, stop outside traffic and wait.
Urban keeps the player and camera still while the other pedestrian walks along
the road. The latter tests a moving NPC policy against the earlier inert policy;
it is a new task, not a controlled single-variable ablation.

## Evidence collected so far

Rural attempt 1 was rejected by Pi without an external correction. It identified
background expansion and camera advance despite a fixed-camera plan, and found
insufficient evidence of ground-relative stepping. Revision 4/time 12 remained
authoritative. Independent inspection agreed. Background feature displacement
was approximately 16.40 pixels at 512x288 (20 homography inliers).

Attempt 2 preserved the background much better: approximately 0.31 pixels of
background displacement (141 inliers). Pi accepted partial forward progress,
explicitly retaining the unfinished road-edge goal and documenting stiff gait
and possible sliding. Codex inspected both timestamped contact sheets and the
whitebox/RGB comparison. Prompt and seed both changed, so this result does not
isolate prompt effectiveness. These image-space metrics are not metric 3D
motion or camera calibration.

Rural finished with three generations: one rejection and two partial accepts.
The fork now has six accepted historical segments spanning 18 seconds, including
four inherited segments. The road-edge goal remains pending. All six segment
checks pass: accepted starting scene, actual prior RGB endpoint, static entities,
revision sequence and saved receipt. Its entire immediate-source history is
unchanged. No root position corrections were applied: accepted metric roots
remain qualitative plan-consistent estimates, not recovered 3D trajectories.

Dense crops at 0.125-second intervals show very little alternating leg motion
in rural attempt 2. Sliding is visibly substantial; the acceptance establishes
coarse progress only, not natural walking quality. `player_motion.jpg` retains
this evidence without interpolation.

Urban generated its 1024x576 clip in about 1166 seconds. The NPC transitions
out of the initial running pose during the opening second, but sustained gait
is weak afterward. The player remains in place. Background displacement is
approximately 0.25 pixels (933 inliers). Dense `npc_motion.jpg` documents the
pose transition followed by weak leg articulation. Pi accepted the segment and
marked the action complete, recording a stationary player and an NPC locomotion
estimate of `[0.2,0,-0.8]` in inferred world axes. This successfully exercised
the optional NPC motion assessment in a real generation/commit. Independent
visual review accepts coarse progression but does not endorse natural-gait
quality as solved. The branch reaches revision 4/time 12 seconds.

The rural accepted montage contains 433 samples over an 18-second world span;
the urban branch contains four accepted segments over 12 seconds. Shared
boundary timestamps are omitted, without blending or interpolation. First-frame
conditioning still redraws some detail: rural's new boundary RGB differences
are approximately 5.26 and 5.17/255, greater than adjacent-frame differences.
Correct reference paths do not imply pixel-identical continuity.

The source-world audit compares all rows of objects, metadata, evidence and
edits against the local pre-trial copies. Both sources are unchanged. The
report keeps the original trials separate from these continuations. Both
branches pass all history-chain checks and both Pi processes exited with code
0. All four GPUs returned to their idle 15 MiB state. No generation remains
running for this follow-up.

## Shared fixes

- `plan` now returns actual video frame count, duration, alignment changes,
  diagnostics and compact actor/camera endpoint state. The saved full files
  remain available. Two existing real-video replay tests passed locally and
  on the server, including the new return-field checks.
- The launcher now describes the actual H3 frame grid: `17*n+5` samples,
  including the initial frame. At 24 fps, 73 samples span 3s and 124 span 5.125s.
  Earlier launch text requested 121 samples for 5s; Pi detected the mismatch
  and chose 3s. The running trial prompt is preserved, not rewritten afterward.
- Explicit depth requests no longer receive the misleading outer metadata
  label `source-derived Canny`. The actual controls are recorded. Earlier
  generation commands and `controls_request.json` remain the evidence for
  their real settings; earlier metadata is not overwritten.
- Export and history audit scripts recognize independent branches. The audit
  compares a branch to its immediate source, including all inherited history,
  rather than only comparing it to the original scene baseline.

## Commands

From the project parent directory:

```powershell
python -X utf8 agentic_world/scripts/run_pi_interaction.py status --branch rural/road_edge
python -X utf8 agentic_world/scripts/run_pi_interaction.py status --branch urban/npc_walk
python -X utf8 agentic_world/scripts/run_pi_interaction.py fetch --branch rural/road_edge
python -X utf8 agentic_world/scripts/render_gta_loop_evidence.py --case rural/road_edge
python -X utf8 agentic_world/scripts/export_gta_accepted_rollout.py --case rural/road_edge
python -X utf8 agentic_world/scripts/report_gta_pi_loop.py
python -X utf8 agentic_world/scripts/audit_gta_pi_sources.py
```

Outputs live under `outputs/gta_pi_loop_20261102/`. Inspect failed attempts as
well as accepted videos; the accepted montage deliberately excludes rejected
generations but preserves the saved history order.

## Next useful experiment

Keep a fixed initial frame, camera, route, seed and generation recipe, and compare
the current articulated whitebox against a coarse actor-volume condition. This
would isolate whether visible proxy limbs encourage stiff motion. It is a
hypothesis to test, not a demonstrated cause. Do not add detailed skeleton
constraints or claim that more retries alone resolve it. A separate same-seed
prompt comparison is needed before attributing the fixed-camera recovery to
wording alone. No additional paid generations were launched for these ablations
in this follow-up.
