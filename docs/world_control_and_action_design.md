# Coarse control and action lifecycle

Date: 2026-10-08. This document describes implemented contracts, not an assertion
that the generated action succeeds. Measured trial results are separate.

## Control boundary

The world owns persistent actor identity, coarse position/extent, intended
trajectory, camera trajectory, action order and completion criteria. It does
not own every pixel. Fur, clothing, particles, lighting and incidental
unrecorded background remain renderer freedom. A visually recurring detail is
still an observation, not independent evidence of a world fact.

`compiler/world_controls.py` emits stable-ID color layout on white, inverse
camera-z depth, proxy Canny and proxy gray as lossless uint8 NPZ arrays. These
are compiled from timed world objects and camera, never from source RGB.
Depth also retains float camera-z and a separate known-pixel mask. Unknown
background is not measured far geometry; its black RGB encoding is a control
format approximation. Sparse depth may therefore be a poor condition.

The recorded GTA prefix only supplies actors, not the static garage. Missing
person dimensions are explicitly inferred category-sized boxes. Layout colors
represent subject identity, not semantic category IDs. No fake skeleton is
generated: Pose requires reliable joints not present in this white model.
HED/MLSD are not separately claimed; they would only re-encode the same rough
geometry rather than add world information.

The Union 2.0 model card specifies color-coded bounding boxes on white, eight
control modalities, CFG=1 and 17*n+5 frame lengths. Arbitrary semantic-ID masks
are not its semantic interface. Full semantic/depth proxy conditioning in CWM
uses a separately trained LoRA and must not be confused with Union RGB control.

Format correction during testing: initial trials incorrectly filled the boxes.
Inspection of VACE `LayoutBboxAnnotator.forward` showed thickness=2 outlines,
while LayoutMask/Track use thickness=3 contours. The compiler now emits outlined
boxes, and the original filled-box artifacts remain as an explicitly invalid-format
negative control, not standard Layout quality evidence.

Sources:
- https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0
- https://github.com/buaacyw/code-world-model/blob/main/INFERENCE.md
- https://github.com/ali-vilab/VACE/blob/main/vace/annotators/layout.py

## Persistent action execution

`scripts/run_action_episode.py` initializes 1-4 important objects from a source
image and a goal. GPT proposes observed 2D boxes and uncertain sizes. Backprojection
creates a coarse world with INFERRED provenance, not metric ground truth.
Original-image DINOv2 crops provide appearance identity features.

An action has ordered phases, per-phase duration, coarse displacement, camera
displacement/target, an English render prompt and an observable completion
criterion. Each phase is one 73-frame (~3.04-second) segment by default; 124-frame
(~5.17-second) segments are supported. Time is not rounded to fictitious 3/5 seconds.
The current primitive motions use smooth interpolation followed by a settling
hold (`motion_end_fraction`), not a physics engine. GPT can choose the 73- or
124-frame horizon per phase. Articulated held props can be identity-only and
excluded from Layout rather than forced into coarse moving cuboids. Initial
`held_by` relations are retained separately in world metadata.
GPT can select a continuous camera move, not just a fixed third-person view.
Unseen geometry and aggressive camera changes remain uncertain.

For every segment:
1. Save current world and a separate candidate timeline.
2. Compile coarse layout, render H3, then release its GPU process.
3. Sample first/middle/last for detector/SAM/DINO and Hungarian assignment; use
   nine ordered review frames for GPT action assessment.
4. GPT judges ACCEPT/REGENERATE and separately phase/action completion.
5. Only completed accepted intended transitions update positions with
   AGENT_CREATED provenance. Unchanged objects preserve their provenance.
6. Preserve raw decision, effective decision, observations, world before/after,
   model inputs and action history.

An accepted but incomplete clip is evidence of partial progress, not permission
to teleport the world to the planned endpoint. The current conservative behavior
keeps coarse geometry at the last completed phase and continues/retries that
phase. Inferring a reliable partial endpoint is an explicit remaining limitation.
No generated object is spawned. Missing detections do not delete world objects.

World, observation SQLite, and action history remain separate. `active_action`
is stored in existing world metadata; no schema migration is introduced.
New actions are blocked while RUNNING. Explicit `--cancel` is supported and does
not change world geometry. A new `--goal` after completion/cancellation retains
the working world and uses a new action serial. Budget exhaustion remains RUNNING with
`user_input_ready=false`, never COMPLETE. `--resume` restores a running action.
Process crashes leave the incomplete segment directory for inspection rather
than silently overwriting its evidence.

For continuity the new action runner applies native `MiniMaxH3AddGuide` at frame
zero in addition to the Ref2VA reference. This is distinct from reference-only
conditioning. Its efficacy must be measured; neither its existence nor a final
frame reused as reference proves seamless multi-clip dynamics. No temporal
latent overlap/Retake34 has yet been implemented here.

## Harness choice

We should reuse mature session/tool infrastructure, but it does not supply world
semantics or correct action completion by itself. Keep the domain contracts
above independent from the chat runtime, then integrate one runtime after this
behavior has a regression set. Do not fork a whole coding-agent product now.

Pi exposes `createAgentSession`, custom tools and persistent SessionManager;
its provider/runtime customization is relevant to the existing gateway. The
current SDK should be pinned before integration, not copied from stale examples.
Disable general filesystem/shell tools for the world director unless a task
actually needs executable world code. Typed world tools remain the boundary.
Codex provides SDK/App Server routes for embedding its agent loop, sessions and
events, but the existing chat-completions gateway is not evidence of compatible
Codex backend support. That compatibility must be tested before choosing it.

CWM's released inference documents BF16 Ref2VA + CWM LoRA, RGB anchor plus
depth/semantic-ID reference, 124 frames, Retake34 overlap and 90 new frames per
continuation, 20 Euler steps, H800/FlashAttention3. This is not a drop-in harness
or a benchmark already reproduced on our INT8/Union/4090 setup. Borrow the
separation of executable state and visual realization, and study continuation;
do not claim parity or superiority without a same-task comparison.

Sources checked:
- https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/examples/sdk/README.md
- https://developers.openai.com/blog/codex-as-a-platform
- https://github.com/buaacyw/code-world-model/blob/main/INFERENCE.md

## Limits of completion judgments

The critic receives the original image for identity, the latest committed image
for continuity, and a compact completed-phase ledger. Comparing every new action
against the original pose caused a real false rejection and has been corrected.
Held props inherit their holder's translation unless explicitly moved; this is
a coarse relation, not articulated contact physics. See `validation_20261008.md`
for measured successes and incomplete trials.

Nine review frames can still miss a footfall, contact, a dodge or a recovery interval.
GPT confidence is not calibrated success probability. A visually convincing
counterattack is not physical collision verification. Health, damage, enemy
death and hidden geometry are not inferred into authoritative facts. Action
order, contact and settling need denser temporal evidence before combat can
be considered reliably evaluated.
