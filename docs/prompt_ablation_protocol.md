# Prompt Ablation Protocol

`reboot_live` is the current regression baseline for actor identity. Its
recorded H3 command and prompt are frozen in
`outputs/identity_repair_20261013/prompt_regression_baseline.json`.
The source is the actual `reboot_live/ablation.json` command, not the earlier
generation request. Its prompt is checked against the saved text verbatim.

Every new run must preserve the baseline first frame, depth sequence, seed,
steps, resolution, control strength, control end, cache setting, kernel,
reference image and all renderer flags. A run may change only the prompt text.
It must use a new output directory and must not commit world history.

The first comparison matrix is deliberately narrow:

1. Baseline prompt unchanged.
2. Remove only the player's second, detailed memory description.
3. Remove only the staff's second, detailed memory description.
4. Remove only the tiger's second, detailed memory description.
5. Append the three architectural memory descriptions, preserving the entire
   original prompt. The baseline has no architecture memory suffix to replace.

The action, boundary and appearance portions retain their original bytes;
no new section labels are injected into H3 text. The preparation script saves
the exact prompt and a line-level diff against the baseline. Do not change
action wording and appearance wording in the same run.

Preparation does not execute H3. Every prepared record states
`PREPARED_NOT_EXECUTED`; video outcomes must be recorded separately. The
historical comparison is checked over the full command array, allowing only
prompt and output changes, including flags not known to the preparation script.

An identity pass requires the original player and staff to remain one
consistent pair throughout the video, with no second proxy-like staff bearer.
Static-world and camera checks remain separate gates. A successful identity
ablation is not a successful world continuation.

## Execution and interpretation

`run_prompt_regression.py` executes the prepared cases in isolated output
directories and checks every argument and each named edit. The v2 preparation
stores the historical comparison command, so an unrelated rewrite cannot pass
by changing both the prompt file and its command record. Six focused tests
cover these checks and malformed architecture blocks.

Run1 uses GPUs 0, 1, 2 and 3 for baseline, player-detail removal, staff-detail
removal and tiger-detail removal respectively. The architecture addition waits
for GPU 0. These are the same GPU model, but the removal comparisons are
exploratory until any observed effect is confirmed on a matched device.
Historical reboot_live used GPU 1. The historical command was preserved, but
its full source/environment was not frozen; the current replay is a necessary
check rather than an assumed successful control.

The removed descriptions are supplemental: the action paragraph still contains
identity attributes. Passing after a deletion would not show that all identity
memory is unnecessary. The architecture case tests the three-description block
as one addition; it does not isolate each architectural object or token length.
All cases use seed 102. Generalization across seeds and scenes is untested.

`review_prompt_regression.py` uses the unchanged camera evaluator and stores
each review attempt separately. It saves historical completion and assessment
evidence, and compares the decoded historical/current baseline frames solely
as a reproducibility diagnostic. Pixel difference is not an identity score.
The reviewer is not supplied treatment labels or the experimental prompt.
No generated candidate advances the persistent world in this diagnostic.
