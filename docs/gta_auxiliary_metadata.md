# Auxiliary source metadata

The new `scripts.enrich_gta_training_metadata` command reads verified byte-offset
indexes and exports one audit sidecar per scene candidate. It does not change
the existing image/video conditions, prompts, source records or acceptance.

Each sidecar contains the historical reference and target frames 0, 30, 60, 90,
119. Camera interpolation selects the two actual client records; their actions
and discrete states remain separate, with the original interpolation metadata.
The matched world capture tick selects the world record. No nearest-time guess
or interpolation of health, task state, actions or relations is performed.

| Source fields | Intended use | Current integration |
| --- | --- | --- |
| Action, agent velocity, health, alive, in_vehicle | Action/state coverage and transition sampling | Indexed records exported |
| Camera, calibration and frame timing | Projection and temporal checks | Existing alignment plus bracket records |
| Subject/scene observations and appearance | Appearance changes and memory-reference auditing | Scoped observations exported; visibility not asserted |
| World agents, peds, vehicles, objects | Dynamic category coverage and state comparison | Recorded lists exported |
| Relation edges, lifecycle events | Association changes and memory invalidation evidence | Recorded fields exported; empty/missing stay distinct |
| Environment and task state | Weather, lighting and task stratification | Recorded values exported |
| Static collision geometry and materials | Background reconstruction and geometric coverage | Existing proxy input; standalone training-sidecar integration still pending |
| Compiler counts and pose sources | Diagnose omitted objects and proxy limitations | Diagnostic metadata exported; counts are not visible-instance labels |

Target-period actions and realized world states are auxiliary supervision/audit
data, NOT historical conditioning. The current loader does not inject these
records into the prompt. A future action-conditioned loader must distinguish
planned user commands from observed outcomes before using them as conditions.

The full-record sidecar is an intermediate artifact (~1.2 MB in the first real
example). Final packaging should retain useful fields and coverage summaries;
do not duplicate all world records blindly for every overlapping window.
Coverage and mixture proportions are not yet validated by these exports.

## Run

```powershell
C:\Python314\python.exe -m scripts.enrich_gta_training_metadata --root E:\GTA_555_complete_rebuild_20261025\available_source_training --indices E:\GTA_555_complete_rebuild_20261025\record_adapter --output E:\GTA_555_complete_rebuild_20261025\available_source_metadata --follow
```

The existing source preparation and candidate export continue independently.
On October 28 the missing-source HEAD request returned HTTP 403 UserDisable.
Source recovery cannot complete until access is restored; approximately
104.5 GB remained in the recovery plan. Local processing remains available.

The exporter now always leaves new pairs VIDEO_QA_PENDING, even if the target
window has inherited positive review. Exact paired-media review reuse remains
available. Previously exported inherited-positive samples still require audit
before final mixture assembly; changing the exporter does not retroactively
certify or rewrite them.
