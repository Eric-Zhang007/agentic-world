# Visual spot checks

These checks supplement, not replace, per-window VLM review and full media /
record validation. They are not a claim of exhaustive human frame inspection.
Paths below are sample IDs under the active release's `visual_review/` or
`dataset/samples/` directory. Original RGB and whitebox contact sheets remain
available there. This document does not grant release approval by itself.

## 2026-10-19

### Yard motion and instance correspondence

`ep_20260922_123638_baf9_agent_000_segment_00003_w005`

Viewed both RGB and whitebox sheets. The red-striped-shirt character remains
near the foreground while the shirtless character runs around the yard and
returns beside him. Their identities remain distinguishable in the sampled
views, including their final overlap. The parked pickup changes apparent angle
with the camera but remains the same vehicle. Coarse upright character proxies
do not reproduce running articulation, as expected for this source format.
Historical reference appearance is usable; it must not freeze historical pose.

### Large interface overlay exclusion

`ep_20260922_123638_baf9_agent_000_segment_00032_w004`

Viewed RGB sheet and reviewer decision. Target tiles 3 and 4 contain large map
menus, and tile 5 has a blurred menu transition. Rejection is justified by
substantial non-world image content, not by the helicopter's large camera roll
or the coarse vehicle representation. Small corner HUD masking would not make
this clip suitable.

### Alley locomotion

`ep_20260923_122317_11a9_agent_000_segment_00007_w005`

Viewed both RGB and whitebox sheets. The historical reference distinguishes the
red-striped-shirt character from the blue/yellow-shirt character. The selected
red-shirt identity is followed into the alley. Coarse foreground location,
camera translation and final turn agree in the sampled views. Whitebox stance
is substantially simpler than RGB running articulation, which is intentional.
This checks a historical appearance reference plus coarse motion, not exact
pose or collision supervision.

### Missing underground scene geometry

`ep_20260923_122317_11a9_agent_000_segment_00027_w005`

Viewed both sheets. RGB shows a furnished metro station with walls, ticket
machines, columns and corridors. The paired whitebox shows the character in
empty white space beneath exterior geometry, with the station enclosure absent.
The character projection alone is insufficient to certify the whole pair.
Rejection for missing major scene geometry is justified, independently of the
intentional coarse character articulation. The review for the nearby tunnel
sample `ep_20260923_122317_11a9_agent_000_segment_00030_w001` reports the same
failure; that second sample's images were not separately inspected here.

## Interpretation

- These examples do not establish exhaustive visibility or long-term memory.
- Buildings and ground remain unreferenced context, without certified static
  instance IDs or appearance-memory edges.
- VLM acceptance and SAM segmentation are estimated labels, not human ground
  truth. See each sample's original review and mask evidence.
- Only the completed release report, full ledgers and delivery verification
  determine which samples actually enter the published dataset.
