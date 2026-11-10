#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/gta_source_rebuild
./render-env/bin/python bootstrap_gta555_render.py --root /mnt/hdd/gta_source_rebuild --inventory oss_objects.jsonl "$@"
./render-env/bin/python run_gta555_full_rebuild.py --config config.yaml --jobs smoke_jobs.json
