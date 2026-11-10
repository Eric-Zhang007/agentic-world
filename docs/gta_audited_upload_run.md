# Audited local delivery

The October 29 run audits all 2,541 non-excluded scene candidates from
`available_source_training` and `additional_local_training`, including earlier
sampled approvals. Previously excluded samples remain excluded. Eight calibration
examples were reviewed separately before starting the batch: known background,
truck occlusion, fence opacity and console-overlay cases.

Paid API review was stopped at the user's request. Remaining reviews were done
directly in the active Codex session, using local historical RGB/whitebox and five
target pairs. Existing API judgments retain their original provenance; direct
reinspection takes precedence. Acceptance requires useful shared static background, coarse
alignment, no major visibility conflict, no obstructive UI, and no uncertainty.
This is sampled visual QA, not exhaustive temporal inspection or model training.
No new external review API calls are part of the remaining delivery work.
Individual decisions and batch usage are retained; they are not applied silently
to older source sample records.

Do not run `finish_gta_audited_delivery.py`: it is the obsolete API-waiting path.
Use `assemble_gta_audited_mixture.py` with `--direct-audit`, followed by
`package_gta_unified.py` and `upload_gta_audited.py` through the existing WSL
HF session. Existing deliveries are preserved.

The completed manifest has 7,085 records: 2,101 scenes, 3,539 people, 1,191
vehicles and 254 mixed-object records. Scene approvals comprise 1,407 direct
visual judgments and 694 retained prior API judgments. Object records retain
their earlier QA; they were not all visually re-reviewed in this pass.
The split counts are 6,646 train, 181 validation and 258 test.

Scene references: 1,100 nearby history, 771 cross-camera passages and 230
same-camera returns. Only 21 scenes have reference age above 120 seconds;
long-gap memory and large viewpoint changes remain sparse (61 above 45 degrees).
The ten alternative training rounds each contain 1,732 records: 866 background,
347 people, 346 vehicles and 173 mixed. Rounds may reuse records across rounds.

Sampling rounds target 50% scenes, 20% people, 20% vehicles, 10% mixed objects.
The scene quota is divided equally between nearby history and return/cross-camera
references when both exist. No repeated IDs within one round; group cycling
limits immediate domination by one site/instance. The full accepted pools are
also provided. Ratios are initial data-sampling settings, not training-validated
optima. New episode 0914_131955 is test, 1007_131919 is train; earlier episode
assignments are preserved. Shared GTA locations may occur in different episodes.

Local state: `E:/GTA_555_complete_rebuild_20261025/AUDITED_DELIVERY_STATUS.json`.
Remote target: public dataset `Isla8899/GTA-memory-training-20261021`, directory
`reviewed_local`. No acceptance subset is uploaded. Final success requires
`UPLOADED_AND_REMOTE_SIZE_VERIFIED`, not merely an active upload process.

## Upload connection recovery

On October 29 the proxy on port 7897 served the Hugging Face API, but Xet
chunk uploads timed out and stalled. Port 10085 did not accept an HTTP proxy
request. Retrying with the existing API proxy and direct Xet connections
produced successful chunk responses. Use both environment variable cases in WSL:

```text
NO_PROXY=localhost,127.0.0.1,.xethub.hf.co,.hf.co
no_proxy=localhost,127.0.0.1,.xethub.hf.co,.hf.co
HF_HUB_DISABLE_PROGRESS_BARS=1
```

This does not bypass the proxy for huggingface.co. File scanning, Xet processing,
and bytes committed remotely are different quantities; do not describe scan
progress as uploaded bytes. Preserve the verified archive and Xet cache when
retrying. The root status retains the obsolete API-review failure as history;
the linked delivery and remote-status files describe the current run.

Upload completed on October 29 at approximately 22:17 China time. Remote
size is 74,614,669,363 bytes, matching the verified local archive. An unauthenticated
public download HEAD followed the CDN redirect and returned HTTP 200 with that
same content length. Verified repository commit:
`c91c2c7f6b4b89b07dc33a3e10cca7a9a6e850a6`.
