# Scene refinement and real continuation, 2026-10-12

## What changed

Earlier camera fitting used only two hard-coded actor identities and did not
recheck static scenery. A camera that matched actors left important architecture
outside the view. The new path jointly reviews static layout and actors, checks
archived paths/endpoints, and stores inferred-geometry revisions separately from
world-time events. The initialization prompt no longer prescribes a particular
species, extra NPC or attack. Initialization now checks static anchor alignment.

The accepted refinement changed nine inferred static primitives. Actor state,
held props, camera, world time and prior action history were unchanged. Original
revision 4 remains in `outputs/arrival_124f_20261010/world.sqlite`. The separate
`outputs/scene_refinement_coverage_20261012/world.sqlite` committed revision 5.

## Actual failures retained

`outputs/scene_refinement_20261012`: first candidate intersected actors. Later
candidates fixed architecture but the initial joint reviewer rejected existing
weapon foreshortening and procedural limb silhouette. No commit.

`outputs/scene_refinement_comparative_20261012`: before/after and side-view review
confirmed no actor regression, but static review rejected an unmatched left
background mass. No commit. This prompted an explicit evidence-backed addition
operation, not automatic insertion of generated extras.

`outputs/scene_refinement_coverage_20261012`: the SAME earlier candidate passed
current static and actor checks plus four historical endpoint comparisons and
all archived actor collision checks. No additions were actually used. This is
important evidence of reviewer variability: this is not proof that adding the
new operation fixed a missing mass. Original judgments are retained. The
remaining left-background coverage ambiguity is not resolved by another ACCEPT.

A gateway timeout interrupted a review. Resume reuses saved proposals/renders
and valid reviews without overwriting them. Raw alignment responses are retained
before schema validation. Required static IDs cannot be silently omitted.

## Real H3 results

`outputs/refined_rollouts_20261012` branches from revision 5. Two real H3 clips:

| Segment | Intention | GPT result | Revision | Sampling |
| --- | --- | --- | --- | --- |
| 00 | Step screen-left about half body width, settle there | ACCEPT, complete | 6 | 657.65 s |
| 01 | Hold actual attained location without repeating/resetting | ACCEPT, complete | 7 | 656.04 s |

Both use 124 frames at 24 fps, 40 steps, depth strength 1, native first-frame
guide, Triton backend, no EasyCache. No new dependencies or weights were added.
The observed time span is 5.125 s per clip. Visible actors were explicitly
CONFIRMED_PLAN; no nonzero position, posture or camera corrections were applied.
Offscreen patrol state remains UNOBSERVED_SIMULATED. Accepted event times and
interpretations were recorded before planning the next segment.

Final world time: 26.5 s. `chain_verification.json` confirms prior history
preservation, next-plan input equals accepted state, corrected endpoint equals
database, and exactly two new accepted action records. GPU memory was released.

Two manually declared background regions in segment 00 and the same regions in
segment 01 showed near-zero accumulated image translation at analysis width 640.
These are diagnostic observations, not world-specific policies or calibrated
3D camera estimates. They do not establish all-background rigidity.

## Inspection

- `accepted_sequence.mp4`: both accepted videos.
- `boundary_continuity.jpg`: prior endpoint, next start, next endpoint.
- `segment_00_accepted_comparison.mp4`, `segment_01_accepted_comparison.mp4`:
  actual video alongside accepted coarse timeline.
- `segment_*/review/`: raw GPT decision, alignment, before/after and history.
- `segment_*/background_evidence/`: optional image-motion evidence.
- `outputs/scene_refinement_coverage_20261012/integrity.json`: revision isolation.

## Validation and remaining work

81 related regression tests plus 3 resume tests passed on the server. These are
not visual evidence. Two accepted clips support this particular sidestep/hold
continuation, not generalization, accurate metric reconstruction, contact, object
drops, unrestricted camera motion or long-horizon stability.

Next useful tests: explicit obstacle routing; a consequential contact/ownership
transition with a representable event/state contract; and transfer of the same
workflow to a distinct scene. Perception remains optional diagnostic CLIs, not a
complete agent skill loader. Reviewer variability needs evidence-grounded
evaluation rather than repeatedly rerunning a review until it accepts.
