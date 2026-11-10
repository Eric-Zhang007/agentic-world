# Combined local candidate sequence

2026-11-10. This combines an image-backed addition, two real later observations,
fresh local perception, world-state propagation and visual review. It is not an
H3 generation run or an accepted-world/history commit.

## Actual sequence

1. Start with the existing inferred urban scene and create the support-constrained
   parking bollard using `propose_image_primitive`.
2. Read actual later RGB frames at relative capture times 4.532 and 9.056 s.
3. Fit camera translation to explicit static landmarks. Lift visible annotated
   actor contact points onto their explicit support plane.
4. Compare each candidate to fresh SAM2 masks, using explicit component groups.
5. Save each preceding scene and candidate separately, then use that candidate
   as the next diagnostic step's starting state. Render both with local Godot.

Output: `outputs/local_combined_maintenance_20261110`.
`comparison.jpg` contains RGB, actual Godot and mask/projection review in three
columns. Both source masks and the combined visual result were inspected.

The seed is an existing inferred scene, not fresh automatic image reconstruction.
Region prompts, contact points and associations were supplied by Codex. No paid
model calls or remote server were used. No behavior between these sparse frames
is reconstructed; root updates do not infer limb articulation or new events.

## Results

`chain_checks.json` verifies exact preceding-candidate propagation and unchanged
static entities, including the newly added bollard. Each frame reviews four
observations corresponding to six of 26 world entities. Unreviewed entities
remain explicitly listed, not implicitly accepted.

| Check | First later frame | Second later frame |
| --- | --- | --- |
| Numeric state update | 8.5 ms | 4.5 ms |
| Player bbox center error | 29.0 px | 30.1 px |
| NPC bbox center error | 28.6 px | 28.5 px |
| Lamp assembly bbox center error | 8.4 px | 8.3 px |
| Booth assembly bbox center error | 62.9 px | 59.1 px |

SAM2 was freshly run in `outputs/local_urban_sequence_sam_20261110`.
Load: 3.805 s. First frame encoding/prompts: 0.465/0.301 s; second:
0.111/0.025 s. The maintenance runner then consumed those saved results and
accurately reports `perception_invoked=false` for its own stage. Timings exclude
manual annotation and rendering. Actual Godot processes completed successfully.

The persistent static scene and roots can be carried through the sequence, but
this does not establish adequate visual reconstruction. Actor silhouette/height
and booth/background geometry remain inaccurate. Standing coarse proxies also
cannot be judged against a running silhouette as if the shapes should coincide.
The candidate sequence was not accepted as a complete world-model rollout.

## Reproduce

```powershell
python scripts/align_sparse_observations.py --scene outputs/local_supported_extension_20261110/scene.json --annotations outputs/local_supported_extension_20261110/sequence_annotations.json --output outputs/new_combined_maintenance
```

The runner's reusable `run_sequence` now accepts optional paired `perception`
and `mapping` fields in each observation. A perception bundle must contain the
same decoded source image, not merely the same dimensions. It writes
`scene_before.json`, `scene.json`, `report.json`, `overlay.jpg`, and optional
`visual_report.json` / `visual_overlay.jpg` per step. No automatic acceptance
threshold was added. Tests cover state chaining, preservation, visual artifacts,
incomplete review inputs and same-size wrong-frame rejection.
