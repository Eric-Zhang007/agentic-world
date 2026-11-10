# Prompt-only regression results

Server root: `/home/star/agentic_world/outputs/identity_repair_20261013/prompt_regression_run1`.
Matching local root: `agentic_world/outputs/identity_repair_20261013/prompt_regression_run1`.
Six actual H3 videos completed successfully. No candidate was committed to the
world. No model, dependency or rendering default was changed.

## Results

All candidates use the recorded first frame and depth, seed 102, 124 frames at
24 fps, 1024x768, 40 steps, strength 1.0 and disabled cache. The action and
boundary text are unchanged. The following assessments use the same camera
reviewer plus chronological manual visual inspection.

| Prompt case | Identity | Camera task | Static world | Overall |
| --- | --- | --- | --- | --- |
| Exact baseline | Supported | Supported | Contradicted | Fail |
| Remove second player description | Contradicted | Contradicted | Contradicted | Fail |
| Remove second staff description | Supported | Supported | Uncertain | Fail |
| Remove second tiger description | Supported | Supported | Contradicted | Fail |
| Append three architectural descriptions | Contradicted | Contradicted | Contradicted | Fail |

The player-removal and architecture-addition samples lose the layered skirt,
show very thin legs and rectangular platform-like feet from approximately
0.417 seconds onward. They continue to exhibit orbit-like parallax; a failed
camera task is not a claim of no camera movement. The failure includes an
abrupt change from the accepted boundary. These particular cases demonstrate
identity deformation, not the previous simultaneous duplication of two staff
bearers. Staff ownership and recognizable tiger appearance largely persist.

Architecture addition preserves every character of the complete baseline and
appends only the three historical building descriptions. It therefore shows
that missing player detail is not necessary for this appearance regression.
It does not isolate the three buildings individually or distinguish semantic
content from prompt length, position or other conditioning interactions.

## Reproduction and controls

The new GPU0 baseline and historical GPU1 `reboot_live` have exactly equal
decoded pixels for all 124 frames (mean absolute channel difference 0).
A further complete baseline on GPU1 also matches the new GPU0 baseline
exactly. This supplies a same-device control for player deletion, which ran
on GPU1. Baseline and architecture addition both ran on GPU0. The staff and
tiger deletion screenings ran on GPU2 and GPU3 respectively.

`verification.json` confirms exact named prompt edits, equality of every other
command argument except output location, unchanged input bytes relative to
the snapshots taken during this run, and unchanged inference script. Recorded
Torch 2.9.1+cu128, CUDA 12.8, ComfyUI commit and model variant agree with the
historical baseline. Full historical dependency and input snapshots were not
available; the observed exact decoded reproduction is the stronger empirical
check for this case. This is not a general determinism guarantee.

The independent camera reviewer was not supplied treatment labels or the
experimental prompt. It saw the common intended action, original boundary,
chronological video samples and planned proxies. Exact GPU1 baseline equality
was checked directly; no redundant semantic API review was needed for the
pixel-identical replay. Review outputs remain assessments, not geometric truth.

The saved `execution_summary.json` covers the five initial cases. The sixth
replay has `matched_device_baseline_gpu1/exit.json`. Historical completion and
review evidence are retained in `historical_evidence/`.

## Artifacts

- `detail_comparison.mp4`: baseline and three deletion cases, aligned 24 fps.
- `architecture_comparison.mp4`: baseline versus architecture-only addition.
- Per-case `generated.mp4` locally, `render/controlled.mp4` on the server.
- Per-case prompts, diffs, actual commands, completion records and review JSON.
- `review_summary.json`, `verification.json`, and historical evidence.

## Consequences

Keep the exact successful prompt as the identity regression reference; do not
replace it with a fresh agent rewrite. This experiment supports protecting and
testing previously successful wording, not hardcoding a universal clothing
incantation. Memory retrieval and prompt admission are different operations:
retrieving a true description does not establish that appending it improves H3.
Do not promote these failed prompt variants to normal rollout defaults.

Next isolate each architectural sentence on the frozen baseline, then test
length/placement controls and another seed. Separately compare the historical
failed action paragraph with the baseline while keeping the identity suffix
fixed. Do not change geometry during these text experiments. Static geometry
alignment remains a separate unresolved task even when identity is supported.

Seven preparation, execution-validation and baseline-evidence tests pass
locally and on the server. These are code checks, not video acceptance.
