# Object-centric data preparation

Derived data lives in `E:/GTA_memory_prepared`, separately from the live
download in `E:/GTA_555`. The root README describes package contents and scope.

Implemented entrypoints:

- `scripts/index_gta_memory_segments.py`: stream frame metadata, split valid
  runs at missing cameras, frame discontinuities or clock-domain changes;
  exclude segments without local paired videos or camera Parquet.
- `scripts/prepare_gta_object_memory.py`: verify actual bracketing camera and
  world-tick joins; interpolate camera rays/poses; project dynamic OBBs;
  estimate dynamic occlusion through ray/OBB nearest intersections; export
  RGB/proxy clips, states, reference frames and review artifacts.
- `scripts/package_gta_memory_pilot.py`: apply an explicit scene-bound manual
  review; validate entity lifecycle/model and historical reference time;
  write object reference-use edges and a portable relative-path sample index;
  create and integrity-check a ZIP.
- `configs/gta_memory_pilot_review.json`: explicit manual annotation for the
  single reviewed vehicle example. It is not a reusable automatic label rule.

Verification: six geometry/contiguity tests pass. A real frame window starting
at 780 crosses the recorded missing-camera interval and was rejected before
creating an output directory. The accepted pilot has 24 RGB/proxy target
frames, 27 camera sample records, 21 world records and one reference edge.
Reference-to-target age is 5.022992634 seconds. The ZIP contains 47 entries
and passed decompression integrity validation.

Review is scoped to visible vehicle exterior. Coarse geometry masks are
never labeled SAM masks or ground-truth segmentation. Dynamic occlusion
does not account for missing static geometry. The 120-frame diagnostic
window shows later camera/vehicle alignment differences and was not promoted
to the reviewed sample. No full dataset completion or training-readiness
claim is made; long-term revisit and state-change samples remain to prepare.

The task is data preparation only. No model weights or remote inference
environment were changed. No training was started. Geometry resource location
is pending; the downloaded tree does not yet contain the referenced shared
`worlds/gta5_scene_v3_20260908` directory.
