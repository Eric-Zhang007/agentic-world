# Restore findings, 2026-10-26

The prior reports incorrectly treated an unchanged status file as evidence of
progress. The WSL extractor was observed in D state waiting in p9_client_rpc
while accessing /mnt/e. It has been terminated and replaced with the native
Windows Python 3.14 extractor scripts/extract_gta555_native.py.

Native extraction writes byte and current-file progress every five seconds,
records exceptions as FAILED, and creates a shard completion marker only after
the archive iteration completes. It resumes completed shards; an incomplete
shard is replayed. It runs independently of the interactive terminal.

Important completeness correction: this local directory contains only 30
non_jsonl shards, with nonconsecutive numbers from 0001 to 0090. Their first
entries were inspected. Most hold different RGB segments; 0090 starts with
data/global_index.sqlite3. The historical package audit describing 21 episodes,
42 views and 1,992 pairs does NOT prove that these 30 local shards contain that
whole package. JSONL/spool coverage and the complete shard inventory must be
established before claiming full restoration or starting all-pair rendering.

The original runner was a placeholder and ignored nested YAML settings. It has
now been replaced by actual frozen-render execution and decoded-frame checks.
Real 21-frame and 1,200-frame reproductions passed. The 1,200-frame result is
pixel-identical to the published proxy. All 1,992 published proxies have now
passed publication hashes and full decoding, so unchanged videos are reused.
See gta555_full_rebuild_plan.md for the currently running source recovery,
index adaptation, proxy sync, preparation and candidate export queues.
Visual acceptance and final mixed-package delivery remain separate unfinished work.

Runtime evidence: E:/GTA_555_complete_rebuild_20261025/EXTRACTION_STATUS.json,
extract_native.log and extract_native_error.log. Read updated_unix and compare
byte counters across observations; a live PID alone is not proof of progress.
