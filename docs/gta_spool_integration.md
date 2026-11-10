# Raw spool and static scene integration

The existing Parquet-only source gate omitted raw camera/world observations.
`scripts/index_gta_spool.py` now indexes the four earlier paired batches and the
unpaired local 1007 batch into a separate output directory. Sources and existing
training archives are unchanged. SQLite stores JSONL byte offsets; payloads stay
in their original files. Keep those source files available and unchanged.

`memory.gta_release.Records` reads both previous Parquet indexes and raw JSONL
indexes through the same camera/world API. A raw observation occurrence is not
an authoritative lifecycle record: the entity lifecycle table is intentionally
empty until a separately verified lifecycle adapter is implemented. Existing
object-memory qualification must not treat these indexes as full lifecycle data.
Observed entity counts and time ranges are available in `observed_entities`.
They are observations, not per-frame visibility labels.

The batch audit reads every frame metadata record and checks camera/world joins
at 120-frame intervals, including episode and clock agreement. This is input
integration validation, not full media decoding or a new training-data release.
Frames without alignment and sampled join failures remain explicit in reports.
1007 has no local proxy/segment manifests, so indexing its raw observations does
not make it a paired training source.

`memory.gta_scene_memory.inspect_frame` follows the frame's recorded `scene_index`
and exposes static placement IDs, episode-scoped memory keys, ground/obstacle
component counts, and renderer entity IDs. These describe candidate membership
or renderer bookkeeping. They do not establish pixel visibility. Source scene
files contain geometry-selection information, component bounds and resource IDs;
they are not necessarily self-contained renderable meshes.

`scripts/index_gta_scene_history.py --manifest <manifest.json> --output <new.db>`
indexes all frames of a segment into frame->scene->placement membership tables.
For a placement, query its recorded candidate frames with:

```sql
SELECT f.frame, f.time_ns
FROM frame_scene f JOIN scene_placement p ON f.scene=p.scene
WHERE p.placement_id=? ORDER BY f.frame;
```

No visible-instance labels are invented from regional candidate lists. Static
buildings/terrain already occur in the proxy render as coarse collision geometry;
the earlier object-memory training release did not supply their instance memory.
The subsequent scene-reference release references entire RGB/whitebox frames.

Example commands from the project directory:

```powershell
python -m scripts.index_gta_spool
python -m scripts.inspect_gta_scene_frame --manifest E:/GTA_555/0921/derived/segments/agent_000/segment_00000/manifest.json --frame 8 --output D:/GTA_spool_integration_20261020/frame.json
python -m unittest scripts.test_gta_spool scripts.test_gta_record_joins
```
