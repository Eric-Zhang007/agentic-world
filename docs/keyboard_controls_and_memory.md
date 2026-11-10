# Keyboard control and H3 memory inputs

## Keyboard and mouse

Run from `agentic_world`:

```powershell
python scripts/serve_controls.py --scene outputs/gta_asset_alignment/surface_motion_scene.json --port 8766
```

Open `http://127.0.0.1:8766`. Start records a prospective whitebox preview:

| Input | Behavior |
| --- | --- |
| W/A/S/D | Camera-relative horizontal movement; diagonals normalized |
| Drag mouse | Camera orbit yaw and pitch; follows player translation |
| Shift | Run, with bounded acceleration/braking |
| Space | Jump, with gravity and ground landing |
| C | Toggle crouching/standing |
| X | Stand, subject to overhead clearance |
| F | Existing held-staff sweep and approximate contact events |

There is no prone/crawl or firearm aiming, projectile, ammunition or damage
implementation. Low guard is not prone. The GTA street fixture has no staff;
F reports an unsupported attack rather than inventing a weapon. The staff
fixture is covered separately by the contact test.

The controller is Python, not GPT. The browser submits input at 24 Hz and
displays the same scene, exported GTA meshes, camera intrinsics and actor
height factors. A slow server slows preview time; it does not silently discard
simulation ticks. There is no local browser physics prediction. Mobile can
display the preview; touch movement controls are not implemented.

Motion uses bounded acceleration, smooth coarse height changes, ballistic
vertical movement, and sampled collision queries. It does not generate
fine skeletal animation. Mouse orbit is direct input, not a camera collision
solver; camera clipping remains possible. Ground obstacle motion stops/slides;
intentional staff contact records `proxy_contact` without inferring damage.
NPCs share the same guard/patrol policy helper as agent-planned actions.

## Recording, compilation and history

Export waits for an H3-compatible sample count (`ticks + 1 = 17*k + 5`). It
does not add a settling phase, alter recorded input, or zero endpoint velocity.
The current implementation finishes jumps, posture transitions and staff
sweeps before export. Intermediate airborne video boundaries remain unsupported.

Each plan saves `scene.json`, `action.json` (raw controls), `timeline.json`,
`endpoint.json`, `diagnostics.json`, and `status.json`. `check_plan` replays all
inputs using the same simulator before compilation and rendering. Preview
does not create or modify a world database. To bind a recording to an existing
world, pass the exported `action.json` to the existing `world plan` operation
with that database. It will replan from that DB's actual scene and revision.
Existing video inspection/reconciliation/commit is still required afterward.

### World-bound browser mode

The browser can now launch the complete Pi/GPT and H3 loop on the H3 host:

```bash
.venv/bin/python scripts/serve_controls.py --db /path/to/independent/world.sqlite \
  --first-frame /path/to/initial_rgb.png --gpu 2 --port 8768
```

Use an independent experiment database. After an accepted rollout, its actual
last RGB frame supersedes `--first-frame`. The keyboard button records exact
input, binds the plan to the current revision, and invokes
`scripts/run_interactive_rollout.py`. The action-prompt button invokes the same
Pi session path with a text goal. Generation and review run asynchronously;
the next input window begins from accepted state after completion.

The agent uses `plan -> compile -> render -> inspect_video -> commit`, with
up to two rendering attempts per request. A final REGENERATE with unused budget
re-enters Pi with the preserved review and unchanged input; acceptance criteria
are not relaxed. Failed render tool calls also consume the retry budget.
Exporting a paused aligned trace submits it immediately. A paused unaligned
trace must continue recording; the UI never appends silent braking ticks.
`world extend` can add prospective
offscreen objects with AGENT_CREATED provenance; it does not alter observed
history. Extension alone is not counted as an accepted video. Each request
saves its prompt, actual Pi events, job completion receipt and generation
artifacts. A stale recording must be refreshed, rather than applied over a
changed world. Invalid corrected endpoints fail before database commit.

For the existing SSH/proxy configuration, run `python scripts/controls_tunnel.py`
locally and open `http://127.0.0.1:8768`. Standalone `--scene` mode remains a
preview/export interface. The group-meeting evidence lives in
`outputs/meeting_demo`, including separate action, keyboard and browser-test
branches. The browser test is a fork, not an edit of either presentation run.

The browser viewport is an interactive preview. H3's whitebox RGB/depth videos
continue to be compiled by the existing Godot renderer, with the original
camera intrinsics and physical geometry.

## Comparison with the supplied training screenshot

| Item | Colleague's screenshot | Current inference path |
| --- | --- | --- |
| Base | H3 Ref2VA | `run_comfy_control.py` loads `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Historical image | `history_1`: paired historical RGB + corresponding whitebox, concatenated in one image | Default no additional H3 memory image; relevant memory is used for agent reasoning/text; selected object crops can be supplied explicitly |
| Image numbering | Picture 1 is history; Picture 2 is target first frame | Picture 1 is current RGB first frame; optional images follow |
| Whitebox sequence | Video 1 is target whitebox motion reference | Native whitebox reference video; separate from ControlNet |
| First frame | Exact target first RGB frame | Current accepted/source RGB first frame; exact anchoring is a separate renderer option |
| Depth | Not shown | Configurable separate depth ControlNet, recent recipe strength 0.5 |
| Prompt | `object_definitions`, `summary`, `attention_analysis`, `detailed_description`, audio fields | Agent-authored segment prompt + image/video binding + start/end constraints + selected identity text |
| Target RGB | Supervised training target | Unknown output to be generated, not an inference input |
| Duration | Screenshot states 5.00 seconds | Actual request frame count; e.g. 73 total frames at 24 fps includes initial frame and 3 seconds of subsequent motion |

The critical difference is the paired memory image, not the section headings.
The current wrapper does not automatically compose and feed a historical
RGB/whitebox pair. It also does not establish that the colleague's newly
trained memory weights are deployed. Copying her prompt while retaining our
image numbering would bind history and first frame incorrectly.

The screenshot's code pane is clipped horizontally; this comparison concerns
the visible input contract and text, not an unseen full prompt. Neither the
screenshot nor code inspection proves the memory-trained model's quality.
No generation prompt or checkpoint was changed by this keyboard-control work.

When integrating the trained memory model, use its actual training loader's
pair layout, image order, preprocessing and tag binding together. Keep current
first-frame state authoritative for changed objects; history is evidence of
prior appearance, not permission to restore an obsolete state.

## CWM reference and verification

Inspected on 2026-11-02: `https://github.com/buaacyw/code-world-model` and its
project page, Code World Model: Coding Agent as World Brain. The public tree
inspected contains `cwm_h3_inference`, examples and deployment documentation;
no keyboard/mouse controller was found in that tree. Our controller is an
implementation against this project's world representation, not copied CWM
controller code. Three.js 0.180.0 is vendored with its MIT license for preview.

```powershell
python -m unittest scripts.test_keyboard_controls scripts.test_action_semantics scripts.test_whitebox_world
python harness/pi/test_plan_gates.py
python scripts/test_controls_browser.py
python scripts/prepare_controls_trial.py --scene outputs/gta_asset_alignment/surface_motion_scene.json --output outputs/keyboard_controls/trial
```

Browser verification saves desktop/mobile screenshots, input movement checks,
WebGL color sampling and replay export evidence under
`outputs/keyboard_controls/browser_review`. These are whitebox/control checks,
not a claim that a new H3 video or accepted video-world loop has passed.

Verified delivery on 2026-11-02:

- 89 related control, geometry, posture, continuity and reconciliation tests
  passed locally and on the server; the existing 5 plan-gate and 2 world-tool
  tests also passed.
- The GTA street input trial has 72 ticks plus the initial frame: 73 RGB and
  73 matching depth frames, 768 by 432. It jumps at 1 second, lands during the
  following second, and ends with 2.2 m/s horizontal velocity and a nonzero
  camera orbit rate.
- Server plan: `/home/star/agentic_world/outputs/keyboard_controls/trial/20261102_233553_803114`.
- Downloaded video: `outputs/keyboard_controls/compiled_review/whitebox.mp4`.
- RGB/depth contact sheet: `outputs/keyboard_controls/compiled_review/review.jpg`.
- Preview does not consume GPT/H3 API calls. This delivery did not run H3.
