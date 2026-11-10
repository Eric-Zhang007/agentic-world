# Accepted-video appearance memory

The evidence rollout runner now uses the existing `AppearanceMemory` store.
This is derived visual memory, not new authoritative object state. Existing
objects, accepted trajectories, ownership and history retain their authority.

## Runtime path

1. Branch the existing world database. Bootstrap references from its latest
   accepted video when that video has not yet been indexed.
2. Sample the accepted video's first, middle and final frames. GPT proposes
   visible existing object IDs, normalized boxes, observed view labels and
   appearance descriptions. Unknown identities and unseen views are omitted.
3. Validate the entire proposal. Store append-only PNG assets and metadata,
   including video, source frame, accepted revision and visible evidence.
   Repeated indexing of the same accepted record is idempotent.
4. Planning retrieves up to six crops, prioritizing actors and held props.
   Within each priority tier, give each object one reference before allocating
   second references. Preserve actor/held-prop anchors before adding scenery. Retain
   an early accepted identity anchor when no original reference exists.
5. The prompt-compiling agent sees the current boundary followed by these exact
   crops as reasoning evidence. H3 does not automatically receive those crops.
   The agent expresses stable identity in words and may use only the actual
   renderer selection's Picture tags in its output prompt.
6. `renderer_references.json` separately records the images actually sent to H3.
   The default is only `Picture 1`, the accepted full-frame boundary, which also
   supplies the first-frame guide. Supplementary references require explicit
   retrieved IDs passed to `compile_prompt(renderer_reference_ids=...)`; they
   must have accepted/original provenance and are renumbered for H3. The main
   evidence runner currently requests none. Historical crops are never temporal
   keyframe guides. This separation was introduced after a real multi-image
   camera-orbit regression; see `reference_ablation_20261012.md`.
7. Video review can compare identity against historical crops. Only after
   acceptance and world reconciliation are new references indexed. A failed
   enrichment is reported as a memory failure, not a rejected or rolled-back
   accepted world. Recover it before continuing the next action.

The installed Comfy node supports nine image references total. Our bounded
agent retrieval uses at most six images, each at most 512 pixels per side,
plus the current boundary. This is an agent-context budget, not a request to
send all seven images to H3. Crops preserve aspect ratio. No model downloads,
training, new dependencies, or changes to the existing Comfy installation are
required. Additional reference tokens can increase generation time.

## Commands

Existing runner, now with memory enabled:

```bash
sh scripts/with_proxy.sh .venv/bin/python scripts/run_evidence_rollouts.py \
  --parent outputs/refined_rollouts_20261012/world.sqlite \
  --output outputs/new_memory_trial --segments 1 \
  --goal "Retain the staff and stay alert at the accepted location."
```

Recover appearance indexing after a failed API call. A bound capture directory
can be resumed; its saved proposal is reused if present. Invalid proposals are
archived with their validation error so retry can obtain a new proposal. Use a fresh directory
for another source or a new proposal. This only indexes an already committed video; it cannot accept a
pending or rejected video:

```bash
sh scripts/with_proxy.sh .venv/bin/python scripts/capture_appearance_memory.py \
  --world outputs/new_memory_trial/world.sqlite \
  --output outputs/new_memory_trial/memory_recovery
```

## Inspectable artifacts

- `initial_memory/proposal.json`: GPT crop proposals and uncertainties.
- `reference_memory/*.png`: immutable image snapshots, referenced by metadata.
- `segment_XX/plan/appearance_memory/request.json`: planning inputs.
- `segment_XX/appearance_memory/request.json`: agent-retrieved evidence and its image labels.
- `segment_XX/renderer_references.json`: actual H3 image selection and Picture tags.
- `segment_XX/render_contract.json`: exact retrieved reference IDs.
- `segment_XX/render/metadata.json`: actual native reference block count.
- `segment_XX/review/appearance_review/`: historical review inputs.
- `segment_XX/review/appearance_capture/`: newly accepted crop proposals.
- `world_meta.appearance_reference_memory`: reference catalogue.
- `world_meta.appearance_captures`: accepted-video indexing ledger.

Database-only branches retain absolute paths to earlier reference assets;
keep the parent assets available. A portable export is not implemented.

## Validation and limits

Server regression tests cover memory admission, crops, duplicate indexing,
early identity anchors, prompt image order, renderer argument order, invalid
references, existing reconciliation, and continuous-motion contracts.

Real trial: `outputs/appearance_memory_20261012` on the server, with selected
artifacts and `generated.mp4` downloaded to the same relative local directory.

- 61 targeted server regression tests pass.
- Bootstrap selected three references: player, staff and tiger.
- A real 124-frame H3 rollout used four native reference blocks (boundary plus
  three crops), a first-frame guide and depth control. Resolution 1024x768,
  40 steps, seed 91, EasyCache disabled. Native generation took 764.9 seconds.
- GPT accepted the stationary alert/glance action. Revision advanced 7 to 8,
  accepted history grew from six to seven entries, and prior entries were
  preserved exactly. Root/camera corrections were empty with explicit visual
  confirmation, rather than silently inheriting an unassessed plan.
- Post-acceptance indexing added six references, giving nine total across two
  accepted videos. This did not create any world objects.
- A real subsequent GPT plan began at revision 8, world time 31.625 seconds,
  and used six identity crops: three newly accepted and three earlier anchors.
  That next plan is uncommitted and its second video has not been generated.
- `memory_audit.json` checks actual native conditioning, image paths, stored
  reference IDs, preserved history and the next plan's actual starting state.
- The final review implementation was also replayed against the same real
  video in `review_replay/`, using a separate database branched from the same
  parent. It received 17 video samples and historical crops at attachment
  positions 19-21, returned ACCEPT, committed revision 8, and completed memory
  capture (`memory_status: CAPTURED`). No second H3 generation was used for this
  review check. The original parent remains revision 7 with six history entries
  and zero appearance references; object count stays 15 in the main branch.

This establishes a real memory read/write cycle. It is not an A/B experiment
showing a quantified improvement in identity retention. Main-agent inspection
used sampled frames; the automated video review also uses timestamped frames,
not continuous human playback. Long absence/reappearance is not tested here.

This version uses GPT boxes, not segmentation or automatic tracking. A thin
diagonal staff can have a large crop containing other objects and background.
View labels and captions are uncertain visual interpretations. Only visible
views are stored; hidden surfaces are not reconstructed. Bootstrap from an
accepted generated video is not independent original-image evidence.

Retrieval is bounded and deterministic, not an embedding search over all
history. It does not yet optimize view choice for a planned camera trajectory
or record state-dependent appearance changes such as damaged clothing. The
old detection ObservationStore and a general conversational memory framework
remain separate. Long absence/reappearance and multi-view identity stability
require further real-video experiments.
