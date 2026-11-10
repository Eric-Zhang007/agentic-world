# Passage reference expansion, 2026-10-20

The previous 45 scene pairs were not an estimate of source capacity. The earlier
miner imposed a 30-second minimum, five-unit camera proximity, 35-degree view
limit, 15-unit leave/return excursion, and 30-second location deduplication. It
also tested only one geometric candidate per target. These choices excluded
useful historical scene references and were not calibrated to yield.

The corrected objective is an earlier RGB/whitebox frame from a prior nearby
passage conditioning a later whitebox video and its RGB supervision. Exact path
intersection, object correspondence, and a long time interval are not required.

## Measured candidate capacity

The existing validated camera index has 7,552 five-second sampled points.

| Camera radius | Minimum time | View angle limit | Target windows with candidates |
| --- | --- | --- | --- |
| 5 | 30 s | 35 degrees | 406 |
| 10 | 10 s | 60 degrees | 2,274 |
| 20 | 5 s | 90 degrees | 4,891 |
| 30 | 2 s | 110 degrees | 5,869 |

These are geometry candidates, not accepted training counts. The index was not
resampled more densely; additional close crossings can still be missed.

## Corrected selection

Try up to eight strictly historical references for each target, preferring
prior return/cross-camera passages, then recent observations within each type.
Reference times are diversified in ten-second buckets. Try 20-unit candidates
first and expand unmatched targets to 30 units. Camera height remains bounded
to avoid matching different floors. RGB background evidence excludes projected
dynamic objects and HUD margins. A failed reference does not discard the target
until other candidates have been tried.

Do not relabel all nearby frames as revisits. `nearby_history` denotes continuous
local context; `same_camera_return` requires a recorded excursion and contiguous
trajectory evidence; `cross_camera_passage` denotes another camera's earlier
observation. Clock/episode crossings are forbidden. Unknown recording gaps stay
separate and do not enter default training.

Target media reuse existing decoded clips where possible, with full decode and
first/last alignment checks. Rejected target reviews remain excluded. Missing
target reviews remain pending until supplemental visual inspection. New-reference background checks are automatic;
the old target review is not a claim that the new reference was visually audited.

## Visual spot checks actually performed

Twelve expanded comparisons were inspected directly, not just their metrics.
The initial four include:

- `ep_20260921_124218_450d_agent_000_segment_00000_f000720`: about 5.3 s historical
  reference, matching port roadway, container wall and cranes, different people.
- `ep_20260921_124218_450d_agent_000_segment_00023_f000720`: about 40.6 s reference,
  matching night street corner and shops with changed traffic.
- `ep_20260921_124218_450d_agent_000_segment_00036_f001200`: about 195 s reference,
  shared parking lot and hillside, changed camera view and vehicle positions.
- `ep_20260921_124218_450d_agent_001_segment_00025_f000480`: about 9.4 s reference,
  shared covered building entrance, wall signs and ground pattern.

Eight additional comparisons cover the warehouse facade and courtyard, indoor
stairwell, gas station from several positions, airport apron, and coastal roof.
They show recognizable shared background despite changed actors and viewpoints.
Images are included under `reference_spot_checks/` in the manifest archive.
Supplemental target-video review is recorded separately in per-batch ledgers;
it checks five RGB/whitebox frames, not every decoded frame.

These checks are not a claim of exhaustive visual verification. The public
manifest must preserve that distinction.

## Mixing

Keep all 4,984 old object samples. Retain all eligible scene samples and expose
their type, time gap, view angle, episode and spatial-bin distributions. Default
experimental rounds use approximate task ratios 50% scene, 20% person, 20%
vehicle, 10% mixed objects. Nearby continuous history is at most half the scene
portion. Sample without replacement within each round and cycle spatial/object
groups. These are unvalidated starting proportions, not an optimal recipe.

No paid API calls or model training are part of this expansion. Existing release
archives remain unchanged. Final counts must come from the resulting manifests,
not the geometry survey or an interim progress counter.
