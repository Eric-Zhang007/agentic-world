import importlib.metadata
import importlib.util
import json
from pathlib import Path
import urllib.request


if __name__ == "__main__":
    for name in ("torch", "transformers", "numpy", "scipy", "pillow", "pyyaml"):
        print(name, importlib.metadata.version(name))
    from transformers import Sam2Model, Sam2Processor, AutoModelForZeroShotObjectDetection
    print("SAM2 and GroundingDINO classes available")
    for repo in ("facebook/sam3", "facebook/dinov3-vits16-pretrain-lvd1689m",
                 "facebook/sam2.1-hiera-tiny", "facebook/dinov2-small",
                 "IDEA-Research/grounding-dino-tiny"):
        try:
            req = urllib.request.Request("https://huggingface.co/api/models/" + repo)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.load(response)
            print(repo, "gated=", data.get("gated"), "revision=", data.get("sha"))
        except Exception as exc:
            print(repo, type(exc).__name__, str(exc))
    print("cached DINO", list(Path("/home/star/.cache/vbench/dino_model").glob("*")))
