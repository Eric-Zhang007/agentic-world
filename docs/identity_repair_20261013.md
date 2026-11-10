# Identity repair and return to the world-continuity task

The October 12 experiments established an input interaction, not a complete
fix: with one exact prompt and seed, adding either three or six crops produced
a thin proxy-like character; boundary-only conditioning retained its identity.
A freshly compiled short prompt subsequently failed even with boundary-only
conditioning. Therefore removing extra pictures alone is not sufficient, and
missing descriptive detail remains a hypothesis rather than a proven cause.

## October 13 experiments

All tests branch from the last accepted revision 8 world. No rejected orbit is
used as a new boundary. Native H3/ControlNet checkpoints and environment remain
unchanged, with 1024x768, 124 frames, 40 steps, seed 102, depth strength 1 and
EasyCache disabled.

- `outputs/identity_repair_20261013/description_only`: exact previously failed
  `memory_consumer_fix` command and prompt, with only the accepted appearance
  descriptions appended. It uses the same original depth file and boundary
  PNG. This isolates the description addition.
- `outputs/identity_repair_20261013/live`: normal evidence rollout runner with
  deterministic replay of the original prepared 90-degree orbit. GPT sees six
  historical crops, selects relevant identities and writes a fresh prompt.
  Selected descriptions are appended intact; H3 receives only the current
  accepted full-frame boundary. This tests consumer wiring in the real path,
  but is not an identical-prompt ablation.

The first job is diagnostic and never opens a world for writing. The second
uses the existing camera audit before world reconciliation; a failed audit
prevents acceptance. Any later continuation must start from actually accepted
and reconciled state, not the planned orbit endpoint.

## Code verification

An omitted `appearance_object_ids` field now fails prompt validation when there
is retrieved memory, rather than silently appending every remembered object,
including an offscreen NPC. An explicit empty list remains valid. Existing
tests cover unknown IDs, duplicates, selected/unselected references, image
renumbering, historical memory preservation and renderer arguments.

67 focused tests passed before these real jobs completed. Numerical/unit tests
do not constitute visual acceptance. Final outcomes and limitations are recorded
after video inspection and independent camera audits.
