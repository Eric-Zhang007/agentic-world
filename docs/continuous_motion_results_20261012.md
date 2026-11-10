# Real Continuous Motion Test, 2026-10-12

## Outcome

Uninterrupted movement **did not pass**. Actual-state reconciliation and
historical-prefix rendering worked, and the failed continuation did not change
the authoritative world. Do not describe this as two successful action segments.

Parent: `/home/star/agentic_world/outputs/refined_rollouts_20261012/world.sqlite`
(revision 7, world time 26.5). Fresh experimental branch:
`/home/star/agentic_world/outputs/continuous_motion_20261012`.

| Segment | Requested | Observed by agent from timestamped images | Decision |
| --- | --- | --- | --- |
| 00 | About 0.4 m left sidestep in 5.125 s; finish moving at 0.08 m/s | About 0.25 m retained travel, late foot relocation, then settling | ACCEPT partial; corrected roots/events and stored stationary velocity |
| 01 | Complete remaining roughly 0.15 m toward the same target over 4.25 s; finish moving | In-place rocking and staff adjustment, no retained target arrival | REGENERATE; no commit |

Distances and velocities are uncertain agent interpretations of a single view,
not measured metric reconstruction. Main-session inspection of frame sheets
also found body/staff adjustments; dense motion and foot-contact correctness
have not been independently certified.

The final branch is revision 8 at 31.625 seconds, player root approximately
`[-1.110539, 0, -0.644689]`, motion `stationary`, velocity `[0,0,0]`.
The source remains revision 7 at 26.5 seconds. The second plan reads the exact
first corrected endpoint and preserves the original destination. Prior history
is unchanged. `motion_verification.json` explicitly reports
`continuous_boundary_supported_by_agent: false`.

## Actual Rendering

Both calls used existing native Comfy H3 Ref2VA INT8, Union2 depth strength 1.0,
1024x768, 24 fps, 40 steps, Triton backend, GPU 3, no EasyCache. No dependency
or weight download was needed. Seeds were 73 and 74.

- Segment 00: 124 frames, single frame-zero guide, renderer wall time 937.05 s.
- Segment 01: 124 raw frames, **22-frame native video guide at index 0**, renderer
  wall time 902.34 s. Kept raw frame 21 onward: 103 continuation samples,
  102 future ticks / 4.25 s. Only the continuation was reviewed. Since rejected,
  even those 4.25 seconds were not committed to world time.
- Guide metadata confirms the real prefix path/count and resolved index.
  Time/count checks passed using actual encoded video, not mocked inputs alone.

The two durations include loading, text encoding, sampling, decoding and process
cleanup. They are not a controlled prefix-speed comparison. Prefix conditioning
preserved the incoming stationary state in this case; it did not cause the
requested future locomotion.

## Implementation and Validation

- Explicit `end_velocity` opts a move into a full-duration Hermite path; previous
  action replay keeps its old semantics. No automatic wait is appended to
  continuous moves. Moving state requires explicit deceleration before waiting.
- Video review requires motion evidence and uncertainty, with dense opening and
  closing samples and older tail images. Partial acceptance stores actual state.
- The real runner loads `skills/render-rollout/SKILL.md` to compile a shared
  world/action contract into a prompt and saves the exact skill used.
- Prefix RGB and depth history are aligned; overlap is removed without changing
  old history. The full generated candidate remains available.
- Fixed inherited-pending-action bookkeeping so completing a parent's action
  does not skip the first newly supplied goal.
- **76 relevant server tests passed.** Local focused tests passed; the encoded
  video test is skipped where PyAV is unavailable. No local PyAV install was
  needed. Both archived parent plans replay identically under the new simulator.

During the run, prompt inspection found incidental two-handed grip and numeric
rig/camera constraints in the first compiled prompt. The skill was clarified
before the second compilation, which allowed grip adjustment and omitted exact
world coordinates. A further compiler change now filters procedural joints,
weapon angle and grip from its boundary summary; that filter passed tests but
was not loaded into the already-running process. Therefore this is **not** a
controlled ablation attributing improvement or failure to prompt wording.

## Artifacts

- `experiment_sequence_labeled.mp4`: both candidates, labeled COMMITTED versus
  REGENERATE. This is an experiment view, not all accepted history.
- `accepted_sequence.mp4`: only the accepted first segment.
- `boundary_half_speed.mp4`, `boundary_continuity.jpg`: experiment seam.
- `segment_00/review/review.json`: actual corrections and partial-action reason.
- `segment_01/review/review.json`: rejection and proposed retry; not committed.
- `segment_01/continuation.alignment.json`: exact overlap bookkeeping.
- `chain_verification.json`, `motion_verification.json`: state chain and motion
  verdicts, kept separate.

Local viewing copies are in
`C:/Users/zjc/omgd/agentic_world/outputs/continuous_motion_20261012`.

## Next Experiments

1. Use a natural, clearly visible travel distance and speed on a clear route,
   without changing this accepted branch's destination retroactively. Compare
   prompt plus coarse root/depth against an explicit coarse walking cycle.
2. Once the first clip actually ends moving, compare one-frame versus 22-frame
   guides from that same accepted boundary with matched settings. Current input
   ended at rest, so it cannot validate moving-boundary preservation.
3. Let the agent request targeted actor/background tracking when endpoint
   movement is ambiguous. Report observability and uncertainty rather than
   treating pixel flow as metric motion or adding scene-specific thresholds.
