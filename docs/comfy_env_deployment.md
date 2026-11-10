# Isolated ComfyUI environment

Verified 2026-10-03. This deployment covers environment readiness only, not
checkpoint compatibility or generation quality.

## Paths

- Source: `/home/star/agentic-world-comfy/ComfyUI`
- Python: `/home/star/agentic-world-comfy/.venv/bin/python` (3.11.15)
- Official repository: `https://github.com/Comfy-Org/ComfyUI.git`
- Detached commit: `95539f56344958339e39b7582a476267d489b0ee`
- Reported ComfyUI version: 0.37.0
- Deployment script: `/home/star/agentic-world-comfy/comfy_env_setup.py`
- Import check: `/home/star/agentic-world-comfy/comfy_env_verify.py`
- Machine-readable verification: `/home/star/agentic-world-comfy/comfy_env_verification.json`
- Package inventory: `/home/star/agentic-world-comfy/comfy_env_packages.json`

Server GitHub TLS clone failed. The exact official commit was fetched locally,
transferred as a git bundle, and cloned on the server. Origin points to the
official repository. The server source working tree was clean after tests.

## Isolation and Dependencies

The new environment has its own site-packages and a `h3_readonly.pth` file
referencing `/home/star/minimax-h3/.venv/lib/python3.11/site-packages`.
It reuses torch 2.9.1+cu128, torchvision 0.24.1+cu128, torchaudio 2.9.1,
transformers 5.18.0, PyAV 18.1.0 and CUDA libraries without copying or replacing
the H3 installation. Do not remove the H3 environment while using this overlay.

New dependencies include ComfyUI frontend 1.53.6, workflow templates 0.11.68,
embedded docs 0.5.12, comfy-kitchen 0.2.35, comfy-aimdo 0.5.5, torchsde,
sentencepiece, aiohttp, SQLAlchemy, alembic, pydantic, kornia and
opencv-python-headless 5.0.0.93.

Tsinghua PyPI mirror lacked a pinned workflow-template media package; official
PyPI was added as a fallback. No HF/model downloads were performed by this
deployment. HF cache was inspected first; `HF_ENDPOINT=https://hf-mirror.com`
is used for future model operations.

The first uv resolution did not recognize packages exposed by `.pth` and began
redundant GPU-library downloads. It was terminated before installation.
The final installer uses pip inside the new environment, which correctly sees
the shared packages. Original H3 package versions were not changed.

## Verification

All of these completed with exit code zero:

```bash
/home/star/agentic-world-comfy/.venv/bin/python \
  /home/star/agentic-world-comfy/ComfyUI/main.py --help
CUDA_VISIBLE_DEVICES='' /home/star/agentic-world-comfy/.venv/bin/python \
  /home/star/agentic-world-comfy/comfy_env_verify.py
cd /home/star/agentic-world-comfy/ComfyUI
CUDA_VISIBLE_DEVICES='' HF_ENDPOINT=https://hf-mirror.com \
  ../.venv/bin/python main.py --cpu --disable-api-nodes \
  --disable-all-custom-nodes --quick-test-for-ci
../.venv/bin/python -m pip check
```

Verified native imports: `nodes.CLIPLoader`,
`comfy_extras.nodes_minimax_h3.MiniMaxH3FunControlNetApply`,
`comfy_extras.nodes_model_patch`, plus cv2, av and torchvision.
`pip check`: `No broken requirements found.`

CPU startup created only ComfyUI's own metadata database. No server was left
running and no GPU inference or checkpoint loading was performed.

## Important CUDA Limitation

At this commit, `comfy/quant_ops.py` explicitly disables comfy-kitchen's CUDA
backend for torch CUDA versions below 13.0. Therefore the shared cu128 setup
passes imports but does not use that optimized backend.

Eager operations remain available. Triton is opt-in using
`--enable-triton-backend`; its GPU execution has NOT been verified here.
Do not equate successful imports with INT8 generation performance or correctness.
Do not upgrade the shared H3 torch to address this warning. If CUDA 13 becomes
necessary, install a separate complete torch stack only in this ComfyUI env.
