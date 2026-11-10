# GTA actual record join audit

Run from the project root with the existing Python environment (NumPy,
SciPy, OpenCV and PyArrow are already available):

```powershell
python -m scripts.audit_gta_record_joins --episode E:/GTA_555/0921 --candidates E:/GTA_memory_prepared/reference_candidates_0921_v1/lifecycle_qualified.jsonl --output E:/GTA_memory_prepared/record_join_audit_0921_v1
python -m unittest scripts.test_gta_record_joins scripts.test_gta_reference_candidates scripts.test_gta_object_memory
```

Use a new output directory for each run. The audit does not change source
Parquet files, previous review decisions or existing archives.

The compact SQLite index scans complete camera and world Parquets. It keeps
duplicate keys so ambiguous joins are rejected, rather than choosing the first
record. Camera keys include episode, agent, client clock domain and sample
sequence; world keys include episode, server clock domain and capture tick.

For each reference and every target frame, the audit requires unique actual
camera records at both interpolation endpoints and a unique actual world
record. Camera server clocks must match frame metadata. Interpolated origins
must be finite and the measured projection-ray basis must be nonsingular.

An object edge survives only if the lifecycle-qualified ID is present at both
camera endpoints for all selected frames with the same registry model hash.
Signed and unsigned 32-bit model hash representations compare equivalently.
Duplicate scene entity IDs do not count as identity evidence.

Outputs:

- `record_index.sqlite3`: derived record index, not authoritative world memory.
- `audited.jsonl`: original candidates plus frame-level join errors and surviving IDs.
- `summary.json`: measured counts and remaining limitations.

This audit is deliberately narrower than a visibility or state audit. Scene
observations can include occluded entities. A successful world join only proves
that the record exists; it does not prove that doors, held objects, appearance,
or other object state stayed compatible. Static geometry, RGB/proxy alignment,
HUD contamination and actual media decode still need separate checks. Surviving
windows remain candidates, not positive training labels.

Do not merge overlapping reference policies into independent footage counts
or distribute alternate cameras of the same episode across train/evaluation.
