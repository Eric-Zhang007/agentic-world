# Pi entry for Agentic World

This uses the actual Pi coding-agent CLI, pinned to 1.4.0. It reuses the existing
gateway configuration at runtime and stores sessions separately under
`outputs/pi_harness`. Credentials are passed through an environment variable,
not command-line arguments or models.json. No global Pi settings are changed.

```powershell
cd C:\Users\zjc\omgd\agentic_world\harness\pi
npm install
C:\Python314\python.exe -X utf8 run.py --session world-work --check
C:\Python314\python.exe -X utf8 run.py --session world-work
```

The interactive command enables Pi's coding tools. On a Linux rendering host,
install Node >=22.19 and this same lockfile, then use its Python environment and
existing config. H3 environments and weights are not copied by this launcher.

For evidence inspection without write or shell tools:

```powershell
C:\Python314\python.exe -X utf8 run.py --session evidence --review-only --prompt "Read outputs/posture_live_20261010/chain_verification.json and explain partial acceptance."
```

Repeat the same `--session` to resume the persisted conversation. `--events`
writes a new JSONL event file; it refuses to overwrite one. `--image` attaches
images directly. The existing render-rollout skill is loaded explicitly.
The context/token limits in models.json are conservative local configuration,
not a measurement of the gateway's maximum. Pi's zero cost fields are unspecified
provider pricing, not evidence that a call was free.

## Verified integration boundary

The November 2 live test used the configured gpt-6-astra endpoint through Pi,
performed two successful read tool calls (JSON and image), and returned a grounded
assessment. The request did not invoke H3 or mutate the world. Session events and
resume evidence are under `outputs/pi_harness/meeting_*events.jsonl`.

The normal launcher now registers `world.ts`. Its `world` tool calls
`world_tools.py`, which makes no model requests. Pi authors action/review JSON,
reads images through its native read tool, and controls the sequence:

```
fork -> status -> plan -> compile -> render -> inspect_video
     -> Pi image review -> commit -> status -> next plan
```

The old experiment runners remain available, but are not used by this workflow.
The existing atomic `commit_video` implementation still owns state/history writes.
Incomplete actions require the stored continuation goal. A repeated identical
commit returns ALREADY_COMMITTED; a conflicting repeat fails. Every new plan and
evidence set has its own directory. Use a new plan directory for a rejected retry.

Run the local real-video replay regression with:

```powershell
python -X utf8 harness/pi/test_world_tools.py
```

It exercises partial acceptance, attained crouch, continuation planning, rejection,
repeat commit and missing-world handling. The old video is paired with a freshly
compiled simulation because the simulator changed since the historical run.
This is a migration replay, not a newly generated H3 result.

New live Pi events: `outputs/pi_harness/migration_events.jsonl`.
Isolated branch: `outputs/pi_migration_20261102/`.
Windows dependency added for actual video decoding: PyAV 19.0.2.

The live migration run completed: Pi decoded/read frames at 0/1/2/3 seconds,
authored its own review, accepted a crouch as partial completion, committed
revision 2 and wrote next_plan from that attained scene. Seven persisted checks
passed in migration_verification.json, including unchanged inherited history.
Run `python -X utf8 harness/pi/verify_migration.py` to repeat those checks.
The next plan remains a prediction; it has not been rendered or accepted.
Relaunching the same session also succeeded: it queried the persisted branch and
correctly reported incomplete crouch and the continuation directory. Pi performed
automatic context compaction. Evidence: migration_resume_events.jsonl.

Compile and render tools require the existing Linux Godot/H3 host. On November 2
direct SSH timed out because it omitted the required SOCKS proxy. Connectivity
was then verified through `../server_exec.py` (relative to the project root),
which uses 127.0.0.1:10085 and existing SSH keys. The host and project are available;
Use this helper for server commands/transfers.
Pi was subsequently deployed with the pinned lockfile and existing project
Python. Fresh GTA trials on November 2 executed initialize/preview, compile,
H3 generation, image review and commit through this tool chain on the server.
Their evidence is under `outputs/gta_pi_loop_20261102`; the HTML index pairs
planned whitebox video with actual H3 video. Trials use only the initial RGB
and fixed action intentions; held-out dataset whitebox images stay local.
The rural trial finished four generations with one rejection and three accepted
segments (9 seconds). Its first action needed a partial acceptance followed by
a stopping continuation. Only two of three user intentions were reached within
the generation budget; lateral gait remained stiff. This is a low-resolution
orchestration result, not a high-resolution quality claim.
The urban and harbor trials each completed three accepted segments (9 seconds)
at 1024x576 and 40 sampling steps. Urban retained a visibly frozen NPC running
pose. An isolated rural continuation reached 12 seconds of accepted history,
but road-edge arrival remained incomplete. Consult trial_summary.json and
independent_visual_review.md for per-action evidence and limitations.
Run Pi on that host using the existing project Python;
the renderer subprocess still uses its own existing ComfyUI environment.

`--prompt-file` supplies a complete task without shell quoting. Model errors in
the final assistant event now produce a nonzero launcher exit code, including
cases where Pi itself exits zero. The gateway worked directly on this host;
the old HTTP proxy at 127.0.0.1:17890 failed TLS. This is separate from the
required local SSH SOCKS transport at port 10085.

Local tool improvements return exact review IDs and actual final-frame paths.
Video inspection returns a timestamped contact sheet and opening/ending images
through the Pi image-content boundary. Validate without API requests:

```powershell
python -X utf8 harness/pi/test_world_tools.py
node harness/pi/test_review_images.mjs
```

These image-return improvements were prepared after the GTA trials started;
their original active harness did not use them. A separate live model probe
received three embedded images in one tool result, and the isolated rural
continuation used that interface. After all trials finished, the improvements
were deployed to this harness. Native bash now inherits the launcher's Python
directory on PATH; the live shell probe returned the project's venv interpreter.
Optional existing-actor motion assessments are accepted alongside required IDs.
The original rejected NPC review was replayed successfully on a temporary fork
on both Windows and the GPU host, without another generation or source mutation.

Session persistence is provided by Pi; world persistence is SQLite. A process
restart does not resume an interrupted H3 subprocess automatically. Inspect its
generation log and output before retrying into a fresh directory.

Paired visual memory is now exposed through `memory_capture` and `memory_query`.
Capture the approved initialization RGB once. Accepted video commits capture the
decoded final RGB and the corrected current whitebox into a horizontal pair.
Render with `db` to select a historical pair by the planned endpoint camera.
The tool binds history as Picture 1, current RGB as Picture 2, and the compiled
whitebox video as Video 1. It checks plan revision/time and current RGB before
generation. Capture errors do not undo an accepted world commit; retry with
`memory_capture`. Default rendering uses epoch-15 LoRA and no depth ControlNet.

`geometry_propose` runs the existing VGGT/perception/compiler stages, or reuses
matching cached outputs. It returns candidate assets and preview images, never
updates a world DB. Inspect and repair the candidate before initialization.
See `docs/geometry_pipeline.md` for parameters and limitations.

Do not use whitebox_cli preview COMMITTED as a video acceptance. Follow the
actual video reconciliation workflow described in world_instructions.md.
