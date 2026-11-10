# Local extension history regression

Run from the repository root:

```powershell
python scripts/test_local_extension_history.py
```

Result: 4 tests passed in 0.737 seconds, exit code 0.

These tests use temporary, real on-disk SQLite databases and production
commit_video, commit_refinement, and world-tools dispatch paths. An accepted
synthetic action creates nonempty history and advances time before extension.
The corrected timeline is archived locally and checked byte-for-byte.

Coverage:

- Evidence-backed static addition preserves all previous entities and IDs,
  accepted history and exact tick; reopening SQLite preserves the result.
- Duplicate IDs, current actor intersections, missing validation, stale
  revisions, and OBSERVED additions are rejected without changing the SQL dump.
- An intersecting edit after a successful addition leaves that addition and
  the complete database unchanged.
- Explicit prospective world extension preserves previous entities and history;
  a subsequent transform request is rejected without database changes.

Scene refinement represents newly interpreted static geometry as INFERRED.
It requires current, joint actor, and historical validation. Explicit world
extension instead requires AGENT_CREATED spawn edits outside accepted coverage.
Neither path treats a newly supplied primitive as a directly measured OBSERVED
fact. The tests deliberately preserve that provenance distinction.

All observations and visual validation judgments here are synthetic fixtures.
This is persistence and conflict-handling evidence, not real visual extension
proof. No APIs, remote machines, servers, or subagents were used.

No core defect was reproduced by these cases. The first test run used a literal
3.0 tick assertion; simulation accumulates 2.999999999999998. The test now checks
exact preservation of the accepted tick, which is the intended invariant.
Core modules and calibration files were not changed.
