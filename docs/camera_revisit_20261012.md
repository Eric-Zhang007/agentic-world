# Large camera motion and revisiting known views

## Public CWM comparison

Primary sources inspected on October 12, 2026:

```text
https://buaacyw.github.io/cwm/
https://arxiv.org/html/2608.25927v1
https://github.com/buaacyw/code-world-model
https://raw.githubusercontent.com/buaacyw/code-world-model/main/INFERENCE.md
https://raw.githubusercontent.com/buaacyw/code-world-model/main/src/cwm_h3_inference/engine.py
https://raw.githubusercontent.com/buaacyw/code-world-model/main/src/cwm_h3_inference/duv.py
https://raw.githubusercontent.com/buaacyw/code-world-model/main/src/cwm_h3_inference/runtime.py
```

The public Mr. Bean fight demonstration was downloaded through hf-mirror.com
from `NTU-yiwen/code-world-model-project-page-videos`, path
`project_page_final/control_world/mr-bean-fight.mp4`. Local copy:
`outputs/cwm_camera_comparison_20261012/cwm_mr_bean_fight.mp4`.
The file has 754 frames at 24 fps (31.42 seconds). Twelve chronological samples
show rear/front/side views and substantial background changes during combat.
Inspection used sampled frames, not continuous playback or calibrated camera
recovery. This selected demo does not prove zero drift under all camera paths.

The released inference implementation differs materially from this prototype:

| Component | Released CWM | Current prototype |
| --- | --- | --- |
| Appearance | Fixed full-scene RGB anchor reused every window | Current accepted boundary plus historical object crops |
| Learned proxy interface | BF16 H3 Ref2VA with dedicated CWM LoRA, 200 adapted modules | INT8 H3 with Union 2.0 ControlNet |
| Conditions | 336x192 metric-depth + semantic-category DUV reference video | Godot grayscale depth through ControlNet |
| Windows | 124 samples, 34 overlap, 90 new frames | 124 baseline; optional 22-frame native prefix, 102 new ticks |
| Continuation | Retake latent mask and re-encoded tail | Native guide batch; different temporal contract |

CWM's DUV uses logarithmic metric depth and semantic categories 0-11. These
are not persistent entity IDs. DUV encoding, packing and the trained LoRA work
together. Sending semantic colors into the current depth ControlNet would not
reproduce CWM. Its 34-frame Retake cannot be substituted directly into the
native Comfy `17k+5` guide batch either. The existing H3 deployment is retained.

## Tests

Parent: `outputs/appearance_memory_20261012/world.sqlite`, revision 8.
Each experiment branches this database. No existing accepted world is replaced.

1. `outputs/camera_orbit_20261012/baseline_roundtrip`: single 5.125-second
   camera-only -90 degree orbit followed by +90 degree return, 62+61 ticks.
   Same target, radius, camera height and FOV; no commanded actor translation.
2. `outputs/camera_orbit_20261012/two_segment_orbit`: 5.125-second -90 degree
   outbound arc, then return from the actual accepted camera in another
   segment. A rejected or inconclusive outbound clip stops the sequence.

The executable camera API now supports `camera_orbit_yaw_deg`, a smooth Y-up
orbit over the full phase. It excludes `camera_delta`, can combine with follow
and outward dolly, and does not rotate actors or edit scene geometry. Tests
verify radius/target preservation, exact round-trip endpoint, unchanged primary
actor roots and invalid inputs. Existing camera behavior remains unchanged
when the new field is absent.

Each trial retains native H3 1024x768, 124 frames, 40 steps, depth strength 1,
first-frame guide, memory crops and no EasyCache. These are different motion
tasks and seeds, not a controlled memory ablation.

## Acceptance criteria

`scripts/evaluate_camera_revisit.py` separates:

- execution of substantial camera motion (not actor turning or zoom);
- return to the original viewpoint, when requested;
- persistent character/prop identity;
- static masonry, paving, stairs and plinth consistency;
- occlusion and reappearance, with unseen surfaces marked uncertain.

No high image-similarity score can make an ignored orbit pass. A first/last
whole-image SIFT diagnostic reports repeatability and matched-pixel displacement
but is not static-only geometry, metric camera recovery or a world truth score.
The GPT audit sees timestamped generated frames and clearly labeled hypothetical
proxy start/middle/end views. New orbit runs require its PASS before acceptance
and world reconciliation.

## Actual results

Both real H3 jobs completed. Neither passed appearance/world continuity.

| Trial | Generation | Observed outcome | Commit |
| --- | --- | --- | --- |
| Fast outbound and return, seed 101 | 807.89 seconds | Substantial orbit-like parallax and reversal, but player armor/proportions and established masonry change by the first nonzero sample. The original appearance/layout does not return. | REGENERATE; unchanged |
| Slower outbound, seed 102 | 804.41 seconds | Substantial parallax, but the same early player silhouette and masonry discontinuities. | CAMERA_TEST_NOT_PASSED; unchanged |

Each file contains 124 frames at 24 fps: the last timestamp is 5.125 seconds,
and the encoded duration is 5.1667 seconds. The slower test's return segment
was deliberately not generated because its outbound segment failed. This is
not a successful two-segment continuity demonstration.

Both GPT camera audits return FAIL; the first run's existing normal acceptance
review independently returns REGENERATE. The `camera_execution` field judges
the complete camera-only transition as CONTRADICTED, while its evidence
explicitly confirms substantial orbit-like movement. This should not be read
as "the camera stayed still." No calibrated video camera angle was recovered.
The second audit's `return_to_start` is inapplicable to that outbound-only clip;
its SUPPORTED label is not evidence that a return was executed.

Verification compared full relevant world snapshots and history records.
Parent and branches remain at revision 8, world time 31.625, with seven
accepted history entries. Before/after snapshots are identical and prior
history is preserved. See `outputs/camera_orbit_20261012/world_preservation.json`.
Both inference workers and experiment runners exited; no GPU inference process
from these tests remains. No dependencies or model weights were installed.

Artifacts, relative to the project root on both local and server machines:

- `outputs/camera_orbit_20261012/baseline_roundtrip/comparison.mp4`
- `outputs/camera_orbit_20261012/two_segment_orbit/comparison.mp4`
- Each local trial folder also contains `generated.mp4`, `samples.jpg`,
  `camera_review.json`, `camera_task_check.json` and `result.json`.
- Server trial folders retain all inputs, prompts, plans and conditions under
  `segment_00/`; camera evidence is under `segment_00/camera_review/`.

The comparison exports show the planned proxy on the left and generated frame
on the right, synchronized at 24 fps. They are 1280x512 H.264 files. This is a
visual inspection aid, not an assertion that the renderer should copy anatomy
or every intermediate movement from the proxy.

## Diagnosis and next experiment

Slowing the orbit did not eliminate the failure in this pair of trials. Seeds
and paths differ, so this is not a clean speed ablation and does not establish
a unique cause. Strong depth-conditioning pressure toward coarse proxy
silhouettes is a plausible explanation for the thin player body. Existing
proxy/static-world mismatch is another candidate. Neither was isolated here.

All six selected memory crops in these trials cover the player, staff and
tiger. Background references are outside that budget. Reference pictures and
explicit identity-preservation prompts therefore exist, but are insufficient
in these trials. Their presence does not demonstrate effective conditioning.

The next useful experiment is a fixed-seed, fixed-path comparison that first
changes only full-scene appearance anchoring, then independently varies depth
conditioning. Evaluate orbit execution and identity together so a frozen view
cannot appear to win. Separately assess the released CWM DUV/LoRA renderer in
an isolated environment against the same world; do not send DUV colors to the
existing grayscale-depth ControlNet. Avoid replacing accepted history or
repeatedly rewriting prompts until one attractive sample appears.

## Reproduction and checks

The numerical task requirement is available as `--camera-excursion-deg 90` in
`scripts/run_evidence_rollouts.py`. It checks the compiled planned camera path
before generation, so an agent cannot satisfy the test by omitting its orbit.
Both actual trials have an independently saved 90-degree compiled-path check.
This validates the plan, not the angle executed by H3.

Focused regression suite on the server: 51 tests passed.

```bash
.venv/bin/python -m unittest scripts.test_camera_revisit scripts.test_continuous_motion scripts.test_video_reconcile scripts.test_camera_reconcile scripts.test_prepared_rollout scripts.test_appearance_rollout

# Use a fresh output path; export refuses to overwrite existing evidence.
.venv/bin/python scripts/export_camera_comparison.py \
  --preview outputs/camera_orbit_20261012/baseline_roundtrip/segment_00/planned_preview \
  --video outputs/camera_orbit_20261012/baseline_roundtrip/segment_00/render/controlled.mp4 \
  --output outputs/camera_orbit_20261012/baseline_roundtrip/comparison_new.mp4
```

Implementation files added: `scripts/evaluate_camera_revisit.py`,
`scripts/export_camera_comparison.py`, `scripts/test_camera_revisit.py` and this
document. Updated: `whitebox/motion.py`, `whitebox/director.py`,
`scripts/test_continuous_motion.py`, `scripts/run_evidence_rollouts.py`.
