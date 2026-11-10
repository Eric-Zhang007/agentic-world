"""CPU-only environment and native H3 node import verification."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path("/home/star/agentic-world-comfy")
sys.path.insert(0, str(ROOT / "ComfyUI"))
sys.argv = [sys.argv[0], "--cpu", "--disable-api-nodes"]
import comfy.options

comfy.options.enable_args_parsing()
import torch
import nodes
from comfy_extras import nodes_minimax_h3, nodes_model_patch
import comfy_kitchen

report = {
    "python": sys.executable,
    "torch": torch.__version__,
    "torch_path": torch.__file__,
    "torch_cuda": torch.version.cuda,
    "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
    "h3_node": nodes_minimax_h3.MiniMaxH3FunControlNetApply.__name__,
    "model_patch_module": nodes_model_patch.__file__,
    "clip_loader": nodes.CLIPLoader.__name__,
    "modules": {},
    "gpu_inference_run": False,
}
for name in ["cv2", "av", "torchvision", "transformers"]:
    module = importlib.import_module(name)
    report["modules"][name] = {
        "version": module.__version__, "path": module.__file__,
    }
try:
    report["kitchen_backends"] = comfy_kitchen.list_backends()
except AttributeError:
    report["kitchen_backends"] = "No public list_backends API; see startup log."
print(json.dumps(report, indent=2, default=str))
(ROOT / "comfy_env_verification.json").write_text(
    json.dumps(report, indent=2, default=str))
