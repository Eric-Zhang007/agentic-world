"""Reuse the installed quantized recipe/model-placement helpers without editing H3."""
import ast
import json
from pathlib import Path
import sys

import torch
from PIL import Image

from h3_quant_single import split_recipe, move
from h3_vision_conv import install_channels_last
from h3_duration import trim_output


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    torch.set_num_threads(8)
    source = Path(request["repo"]) / "examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Ref2VA.py"
    tree, split = split_recipe(source)
    ns = {"__name__": "__agentic_h3__"}
    exec(compile(ast.Module(body=tree.body[:split], type_ignores=[]), str(source), "exec"), ns)
    factory = ns["ModelConfig"]

    def model_config(**kwargs):
        pattern = kwargs.pop("origin_file_pattern")
        model_id = kwargs.pop("model_id")
        root = Path("/mnt/hdd/MiniMax-H3-Quant" if model_id.startswith("Comfy-Org/") else "/mnt/hdd/MiniMax-H3")
        paths = sorted(root.glob(pattern))
        if not paths:
            raise FileNotFoundError(root / pattern)
        path = str(paths[0]) if len(paths) == 1 else [str(p) for p in paths]
        if pattern.startswith("diffusion_models/"):
            for key in ("onload_device", "preparing_device", "computation_device"):
                kwargs[key] = "cuda:1"
        return factory(path=path, **kwargs)

    ns["ModelConfig"] = model_config
    exec(compile(ast.Module(body=[tree.body[split]], type_ignores=[]), str(source), "exec"), ns)
    pipe = ns["pipe"]
    install_channels_last(pipe.text_encoder)
    original = pipe.model_fn

    def model_fn(*args, **kwargs):
        with torch.cuda.device(1):
            result = original(*move(args, "cuda:1"), **move(kwargs, "cuda:1"))
        if not all(torch.isfinite(v).all().item() for v in result):
            raise RuntimeError("Nonfinite denoiser output")
        return move(result, "cuda:0")

    pipe.model_fn = model_fn
    refs = []
    if request.get("reference_video"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from renderer.reference_video import read_reference
        reference, metadata = read_reference(request["reference_video"], request["frames"],
                                             request["fps"], request["width"], request["height"])
        refs.append({"type": "video", "video": reference})
        print(json.dumps({"reference_metadata": metadata}), flush=True)
    if request["first_frame"]:
        refs.append({"type": "image", "image": Image.open(request["first_frame"]).convert("RGB")})
    with torch.inference_mode():
        video, audio = pipe(
            prompt=request["prompt"], references=refs,
            height=request["height"], width=request["width"], num_frames=request["frames"],
            num_inference_steps=request["steps"], cfg_scale=1.0, seed=request["seed"],
            ref_image_short_edge=480,
        )
    args = trim_output(dict(video=video, audio=audio, output_path=request["output"],
                            fps=request["fps"], audio_sample_rate=32000), request["frames"])
    ns["write_video_audio"](**args)
    print(json.dumps({"output": request["output"], "delivered_frames": len(args["video"])}), flush=True)


if __name__ == "__main__":
    main()
