# Evidence-driven coarse scene refinement

The actor-only camera fit used in earlier experiments did not constrain static
landmarks. It could align two actor projections while placing visible buildings
outside the view. Actor agreement alone is therefore insufficient acceptance
evidence for an initialized scene or a revised camera.

`whitebox_cli.py create` now checks key static anchors against the source image
in addition to the existing action and actor review. The builder prompt no
longer mandates a tiger, an extra NPC or a sweep, and explicitly distinguishes
camera forward from world +Z.

For an existing accepted world, `scripts/refine_scene_from_evidence.py` creates
a separate SQLite branch. The agent can propose existing inferred static
primitive transforms, explicitly evidenced missing static primitives, and camera parameters. It cannot silently change actors,
ownership, terrain, IDs, observed geometry or accepted history through this tool.

Each candidate is rendered and checked against the actual accepted reference.
Static alignment and actor alignment must both pass. Every archived actor sample
is checked for collision with revised obstacles; previous accepted endpoint
images are reviewed against re-rendered historical actor states. These are coarse
checks, not contact physics or a guarantee of all hidden intermediate events.

A successful change appends `whitebox_representation_revisions`, including the
before/after interpretation, visual evidence, uncertainty, revision numbers and
artifact paths. It increments the representation revision without advancing
world time. `whitebox_history` and its archived scenes remain unchanged. Consumers
can distinguish a correction of an inferred model from a building moving in the
story. Old scenes are retained as earlier estimates, not secretly rewritten.

```bash
cd /home/star/agentic_world
sh scripts/with_proxy.sh .venv/bin/python scripts/refine_scene_from_evidence.py \
  --parent outputs/arrival_124f_20261010/world.sqlite \
  --output outputs/scene_refinement_20261012
```

Use `--resume` only for an unfinished, uncommitted run. Completed proposals,
renders and reviews are reused; a changed world or historical ledger prevents
resuming. Raw alignment model responses are saved even when schema validation
fails. No failed audit implies acceptance.

`scripts/run_evidence_rollouts.py` consumes a branch and arbitrary user goals.
It plans from the current database, reviews compiled conditions, generates H3
video and reconciles accepted actual state before the next plan. Partial actions
retain the outstanding goal. Rejected conditions/video stop the bounded trial
for diagnosis rather than adding scene-specific prompt exceptions automatically.

## Limits

- Single-view depth and occluded geometry remain inferred.
- Visual review is fallible and currently uses sparse frames.
- Historical endpoint checks do not cover every unobserved interaction.
- Collision checks are existing coarse primitive checks, not a simulator.
- Geometry edits and explicit missing-object proposals are limited to inferred
  boxes/stairs. Terrain revisions, dropped props and ownership transitions need
  explicit operations and corresponding evidence.
- No generalization claim follows from one courtyard; the generic executor and
  arbitrary-ID tests remove specific assumptions but do not replace new scenes.
