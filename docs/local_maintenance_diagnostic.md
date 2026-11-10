# Local Maintenance Diagnostic

Date: 2026-11-08. Offline execution only; no server or external API used.

## Changes

- `whitebox/scene_refinement.py`: camera proposals update supplied fields instead
  of replacing the complete camera. Partial FOV corrections preserve position,
  target and optional camera state.
- `whitebox/camera_reconcile.py`: validate corrected cameras on copied frames
  before publishing them. A late invalid camera no longer leaves earlier frames
  partially modified. Reject empty frames, invalid FPS and nonchronological or
  nonfinite timestamps before interpolation.
- `whitebox/world.py`: reject blank and non-string persistent identities before
  uniqueness and reference validation.
- `scripts/test_local_world_maintenance.py`: reuse local database/scene fixtures
  and existing refinement tests; add a multi-step addition/revision/simulation
  case, camera failure atomicity and invalid identity coverage.

## Evidence

Executed from `C:/Users/zjc/omgd/agentic_world`:

```text
python -m unittest scripts.test_local_world_maintenance scripts.test_scene_refinement scripts.test_camera_reconcile scripts.test_whitebox_world
Ran 50 tests in 3.326s
OK
```

The count includes inherited refinement cases and their explicit suite run;
it does not represent 50 distinct scenarios.

The multi-step case commits a new inferred primitive, updates its position in
a second revision, verifies database round-trip equality, checks Y-up position
`[9,3,8]` becomes database `[9,-8,3]`, checks the first revision is unchanged,
checks world time stays zero, and runs a subsequent wait simulation containing
the new identity. Existing cases cover duplicate/unknown identities, stale
revision rejection, historical actor collisions, archived timeline preservation,
camera continuation and validation failures without database commits.

## Scope

No edits to `video_reconcile.py` or `motion.py`. No rendering or visual evidence
was obtained: validation labels in these tests are synthetic fixtures, not a
claim of real image alignment. Historical file preservation is exercised by
the existing local refinement suite. The workspace has no accessible Git root
at this directory, so Git-based attribution/diff checks were unavailable.
