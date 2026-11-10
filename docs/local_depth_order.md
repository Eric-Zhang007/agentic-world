# Local depth ordering

Run from the repository root:

```powershell
python scripts/test_depth_order.py
python scripts/diagnose_depth_order.py --nearer-is-larger
```

Use `--no-nearer-is-larger` for the opposite explicit convention. This reruns
the comparison and overwrites diagnostic artifacts only. No inference, API,
network, server, world updates, or metric depth conversion is performed.

Inputs are the fresh `outputs/local_fresh_depth_20261109/00_urban/relative_depth.npy`,
the scene `outputs/local_landmark_fit_20261109/urban/scene.json`, and the source
`outputs/gta_pi_loop_20261102/urban/input.png`. Source and array sizes must match.
The four half-open integer rectangles are manually selected interior samples
after RGB inspection, not detections, segmentation masks, or whole-object support.
The generated overlay was also visually inspected.

`perception.depth_order.compare_depth_order` accepts arbitrary entity-to-box
mappings and requires explicit `nearer_is_larger`. It does not mutate inputs.
Finite pixels contribute to the median; total and finite counts are reported.
Missing, invalid, empty, entirely nonfinite regions, missing entities, and
nonpositive/nonfinite camera depths produce unavailable comparisons. Exact ties
are unresolved, not agreements. There is no calibrated uncertainty threshold.

For pair A,B, predicted margin is median(A)-median(B) when nearer is larger,
and its negative otherwise. World margin is camera_z(B)-camera_z(A), reusing
`whitebox.image_alignment.camera_coordinates` unchanged. Positive means A nearer
for both. Opposite nonzero signs indicate disagreement. The two margins have
different units and must not be subtracted or used as metric calibration.
World depths use entity origins, not ray/surface intersections; actor roots and
box centers differ. This is a limited proxy consistency diagnostic.

## Urban result, 2026-11-09

Nearer-is-larger: six resolved pairs, zero disagreements. Both orderings are
player, lamp_base, hydrant, npc (near to far).

| A | B | Predicted margin | World margin | Disagreement |
|---|---|---:|---:|---|
| player | npc | +4.034254 | +6.488095 | false |
| player | hydrant | +2.261834 | +4.199072 | false |
| player | lamp_base | +1.929623 | +3.393864 | false |
| npc | hydrant | -1.772419 | -2.289023 | false |
| npc | lamp_base | -2.104631 | -3.094231 | false |
| hydrant | lamp_base | -0.332211 | -0.805207 | false |

Artifacts in `outputs/local_depth_order_20261109`: `annotations.json` records
manual region evidence, `manual_regions.png` shows the patches, and `report.json`
records predicted-order evidence and per-pair signed margins. Agreement is not
ground-truth accuracy or acceptance of world geometry. Small manually chosen
patches do not establish visibility or order for other objects or surfaces.

Three unittest cases pass, covering both signs, missing/nonfinite/empty regions,
invalid bounds, missing entities, partial finite samples, ties, and input immutability.
