# Server gateway diagnosis

Server tests used gpt-6-astra, the current configured relay token, Responses,
HTTP(S) proxy http://127.0.0.1:17890 and disabled request compression.
No secret is included in this report.

| Test | Outcome | Input / cached input tokens |
| --- | --- | --- |
| Fresh session, shell command and tool return | TOOL_ROUNDTRIP_OK | 20294 / 9984 |
| Original experiment session, same shell diagnostic | TOOL_ROUNDTRIP_OK | 348610 / 279168 |
| Fresh session, source image inspection | IMAGE_ROUNDTRIP_OK, accurate foreground description | 21404 / 9984 |

Token usage is the CLI turn total, not proof of a single request's context size.
The image exec event stream does not expose an image-tool event; its session
rollout should be inspected before treating the success text alone as tool proof.

The environment key is passed through subprocess env as RELAY_USER_TOKEN and
the provider explicitly selects that env_key. This is not a missing-key issue.
The previous 503 failures are real but do not establish a permanent capacity
failure, broken tool return or an inherently unresumable old session. All three
minimal tests now complete. Old-session reuse carries substantially more input.

Actual client versions differ: local 0.166.0, server 0.155.1. Local also configures
a model catalog; the server previously warned about fallback metadata for sol.
No controlled version comparison was performed; neither difference is a proven
cause. Do not change version, proxy, model and task simultaneously and claim a fix.

Next: execute a bounded real scene task in a fresh session, reuse geometric files
instead of the large transcript, then extend the same session while tracking
per-turn usage and failures. Keep the old session intact for diagnosis.

Server logs in /home/star/agentic_world/outputs:
- codex_fresh_tool_probe_20261105.jsonl
- codex_old_tool_probe_20261105.jsonl
- codex_image_probe_20261105.jsonl
