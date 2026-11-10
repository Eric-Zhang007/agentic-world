# Object-centric memory release contract

The deliverable is the final `dataset` directory and all ZIPs listed by
`DELIVERY.json`. An intermediate sample, a successful unit test, or a processing
completion marker alone is not a release. The finalizer verifies all samples,
reject accounting, split isolation, and ZIP CRCs before writing DELIVERY.json.

## Sample inputs and supervision

Each sample has 120 target frames, normally 24 FPS at 1280 x 720. Both target
streams use the same decoder-index cadence. Original acquisition timestamps,
camera interpolation brackets, and server ticks remain in frame_state_index.json.
Container timestamp irregularities must not introduce a one-frame duration offset.
No frame interpolation or duplication is used to normalize container timing.

Training inputs are the target coarse whitebox video, a strictly historical RGB
and whitebox frame pair, an object-reference prompt, and explicit reference-use
edges. Target RGB is loss supervision only. The caption is authored using target
video during annotation and remains audit-only; it is never an input prompt.
Prompts contain only deterministic reference correspondence instructions.
QA sheets and visual_review.json also contain target information and must not be
fed into the conditioning path. The supplied dataset.py keeps these separate.

Apply valid_pixels.png to RGB loss. The loader also zeros these regions in the
historical RGB frame. Stored images remain original evidence. Estimated SAM masks
are optional object features, not human segmentation labels. DINO embeddings are
auxiliary audit features and are not default conditioning.

## Identity, time, and state

Object identity is episode_id plus dataset_entity_id, qualified against the
entity lifecycle. Model hashes and character IDs are not instance identities.
All target frames must have resolvable camera/world records and a projectable
candidate throughout the selected interval. Projectability is not proof of
visibility: eight target RGB frames and the reference are reviewed separately.

Each edge records observed_at_ns, used_at_ns, used_until_ns, reference_id, the
using sample, lifecycle, and an appearance observation revision. The timestamp
is the source's estimated server time. A revision means a new exterior observation,
not proof that every hidden property remained unchanged. Historical pose, position,
door state, occupancy, and damage must not be imposed on the target frame.
The raw edge ledger retains full lifecycle records for auditing, including later
despawn information. It is not a conditioning schema. dataset.py explicitly
selects allowed identity/reference fields and excludes lifecycle and QA text.

## Measured scope

- Candidates must be at least 24 pixels on both projected box axes.
- At most four largest candidates are visually reviewed per window.
- References are 120 source frames before the target start.
- Windows are non-overlapping within each source video.
- Whole episodes, including both camera views, stay in one split.
- Shared GTA map scenery may occur across episodes. This is not a novel-map test.
- Static buildings and ground have no reliable instance memory labels in this
  release because the referenced static geometry is absent. They remain context.
- No claim is made of exhaustive all-object visibility, exact articulated pose,
  long-term revisit quality, or successful H3 fine-tuning.
- References with uncertain exterior changes (for example, poorly visible
  vehicle damage during impacts) are excluded conservatively. Appearance
  observation revisions identify reference snapshots; they are not certified
  event labels or an exhaustive appearance-change annotation.

## Reproduction

Processing scripts are included under processing_code with requirements-data.txt.
They require the original GTA files, a vision-capable annotation endpoint, and
SAM2/DINO inference weights. Credentials and weights are deliberately not packed.
Local release processing used a separate imageio-ffmpeg 0.4.9 executable for NVENC,
selected through FFMPEG_BINARY, because the installed newer FFmpeg requires a
newer NVENC driver API. Source decoding/ffprobe and final checks use the system
FFmpeg and OpenCV. The H3 environment is not modified.

The reader needs only NumPy and OpenCV. Extract the manifest ZIP and all episode
ZIPs into one directory. `GTAObjectMemoryDataset(root, 'train')` returns separate
conditions and supervision dictionaries. A model-specific temporal adapter may
be needed when the trainer expects 4n+1 rather than 120 frames; apply exactly the
same temporal selection to RGB, whitebox, projected boxes, and state indices.
