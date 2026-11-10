# Tool-assisted rural correction

## Verified on 2026-11-04

Pi registers `world geometry_propose`. Its pipeline runs VGGT reconstruction,
GroundingDINO detection, SAM2 segmentation, geometry compilation and Godot preview.
Matching-image cached results can be reused. Candidate previews are returned as
images to the agent. This is proposal generation, not automatic alignment approval.

The real Pi session `rural-geometry-assisted-20261104` called geometry_propose
with the existing same-image reconstruction and merged perception caches. It then
read the original reconstruction report. The next model request failed:

```text
403: {"message":"insufficient balance","type":"billing_error"}
```

No corrected scene or completed before/after evaluation exists for this session.
No world was initialized, no historical database was modified, and H3 was not run.

Server artifacts:

```text
/home/star/agentic_world/outputs/rural_geometry_assisted_20261104/proposal/
/home/star/agentic_world/outputs/rural_geometry_assisted_20261104_events.jsonl
```

The initialization instructions now require inspecting perception evidence and
re-rendering revised proposals. This is an agent instruction, not a new hard gate
certifying image alignment. Existing geometry/schema checks do not certify that
the reconstructed scene matches the RGB.

Local regression verification:

```text
python -m unittest scripts.test_geometry_pipeline scripts.test_geometry_category_mapping -q
Ran 21 tests; OK.
```

## Resume after gateway balance is restored

Run on the server from `/home/star/agentic_world` with the existing Node directory
on PATH. Preserve the original event log by writing a new event file:

```bash
env PATH=/home/star/.nvm/versions/node/v22.23.2/bin:$PATH \
  PYTHONPATH=/home/star/agentic_world \
  .venv/bin/python harness/pi/run.py \
  --session rural-geometry-assisted-20261104 \
  --prompt-file configs/rural_geometry_assisted_resume.txt \
  --events outputs/rural_geometry_assisted_20261104_resume_events.jsonl
```

The resume task reuses the completed proposal and asks the agent to produce
scene revisions, explicit landmark correspondences, measurements and a visual
comparison. It must report remaining alignment and support failures honestly.
