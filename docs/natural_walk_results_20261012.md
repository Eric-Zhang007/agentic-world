# Natural Walking Test, 2026-10-12

Result: no accepted moving endpoint. No second video was generated because the
prerequisite for testing moving-boundary continuation was not met.

The experiment starts from the unchanged revision-7 parent at world time 26.5,
`outputs/refined_rollouts_20261012/world.sqlite`. It does not overwrite the
earlier partial-sidewalk experiment or its outstanding destination.

## Conditions and Repairs

1. `outputs/natural_walk_20261012`: GPT planned 1.3 m travel and 0.65 m camera
   translation. Review flagged cropping despite returning usable. The main
   session interrupted its own runner before H3, retaining the uncommitted plan.
2. Added general `camera_follow_actor`: camera position and target follow the
   actor's simulated displacement, rather than an independent easing curve.
   `outputs/natural_walk_follow_20261012` was rejected with sparse evidence.
3. `outputs/natural_walk_evidence_20261012`: replayed the SAME uncommitted plan,
   after state and deterministic replay validation, with nine chronological
   images and clarified weapon-contact semantics. Walking and staff retention
   were now recognized; foot framing was still rejected. This is an evidence
   revision, not an independent model-performance sample.
4. Added `camera_distance_delta`, a continuous outward dolly preserving the
   initial camera. `outputs/natural_walk_dolly_20261012` planned the same
   `[-1.2,0,-0.5]` m route with player following and 0.7 m gradual pullback.
   Nine-frame preview review passed with all body parts visible.

## Real H3 Result

The final candidate used the existing H3 Ref2VA INT8/Union2 depth setup:
1024x768, 124 frames at 24 fps, 40 steps, seed 81, depth strength 1.0,
Triton, GPU 3, no EasyCache. No model or dependency download was needed.

The rendered character visibly lifts and alternates feet, with natural staff
adjustments and changing framing. The agent could not establish the intended
retained ground-relative traversal or continuing endpoint velocity. It returned
REGENERATE, noting the ambiguity between camera motion, articulation, and root
travel. Zero velocity and route-canceling corrections in the rejected review
are provisional interpretations, NOT committed facts or measured motion.

The original and experimental worlds remain revision 7 at time 26.5. The state
chain verifier reports zero accepted segments and unchanged prior history.
Dense inspection of the last second measured image flow in all 24 adjacent
pairs; this does not isolate actor root velocity or resolve 3D camera ambiguity.
No agent re-review was performed to turn those flow measurements into acceptance.

## Validation and Files

81 relevant server tests passed after the changes, including constant relative
follow framing, smooth outward dolly, invalid camera input, exact plan reuse,
stale-world refusal, tampered-trajectory refusal and existing reconciliation.

Main server output: `/home/star/agentic_world/outputs/natural_walk_dolly_20261012`.
Local review copies: `C:/Users/zjc/omgd/agentic_world/outputs/natural_walk_dolly_20261012`.

- `experiment_sequence_labeled.mp4`: actual rejected video with status label.
- `samples.jpg`: chronological stills, not proof of continuous motion.
- `segment_00/review/review.json`: rejection, uncertainty and uncommitted estimates.
- `tail_motion/evidence.json`: optional dense image-flow evidence and limitations.
- `chain_verification.json`, `motion_verification.json`: history and motion results.

Next useful comparison: hold camera and route conditions comparable while
testing coarse root-only versus explicit step/stance progression. A visibly
moving accepted endpoint is required before comparing one-frame and historical
video guides. The current procedural gait is not a recovered skeleton or
ground-contact simulator, and the current reviewer still receives timestamped
images rather than continuous video understanding.
