# Source delivery

This repository contains the GPT/world tools, whitebox compilation, H3 adapters,
perception, reconciliation, memory, skills and diagnostic scripts.

## Setup

Install requirements.txt in an isolated Python environment. Configure paths in
configs/default.yaml for the existing H3 deployment and supply your own API key.
Weights, H3 upstream source, Godot binary, input datasets and accepted episode
artifacts are external dependencies and are not bundled. Do not reinstall an
already working H3 environment. Historical configuration paths need adjustment.

## Entry points

- scripts/run_evidence_rollouts.py: accepted episode -> plan -> H3 -> review.
- scripts/run_action_episode.py: image/action-driven episode.
- harness/pi/world_tools.py: world tool implementation, retained for reuse even
  when the invoking agent is Codex. The older interactive runner still uses Pi;
  this snapshot does not claim that runner has been migrated to Codex.
- skills/: agent-facing tool workflows.
- renderer/: existing deployment adapters.

Read the matching docs and inspect each command with --help before running.
Outputs and accepted worlds must be supplied separately; historical docs refer
to experiments that are not included in this source delivery.

## Verified in this delivery

48 local tests passed for geometry source mapping, geometry pipeline, world
maintenance, camera reconciliation, observation alignment and sparse review.
The evidence-rollout CLI imports and --help run successfully.
The GPU server is unavailable: a fresh GPT -> H3 -> accepted state -> next clip
run has NOT been verified for this package. Local tests do not establish that.

Credentials are replaced only in this release copy; original working files are
unchanged. Large binaries, generated outputs and dependency directories are
excluded.
