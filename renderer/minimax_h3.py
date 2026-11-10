import os
from pathlib import Path
import subprocess
import time

from common import save_json


class H3Renderer:
    def __init__(self, config):
        self.config = config

    def render(self, prompt, first_frame=None, depth_sequence=None, semantic_sequence=None,
               output_dir=None, seed=None, reference_video=None):
        if depth_sequence is not None or semantic_sequence is not None:
            raise ValueError("Installed Ref2VA has no depth/semantic ControlNet; use image reference")
        if reference_video is not None and first_frame is not None:
            raise ValueError("Choose one reference modality for this baseline: image or video")
        if reference_video is not None:
            from renderer.reference_video import read_reference
            _, reference_metadata = read_reference(reference_video, self.config["frames"],
                                                    self.config["fps"], self.config["width"], self.config["height"])
        else:
            reference_metadata = None
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        request = {**self.config, "prompt": prompt, "first_frame": first_frame,
                   "reference_video": str(Path(reference_video).resolve()) if reference_video else None,
                   "output": str(output / "generated.mp4"),
                   "seed": self.config["seed"] if seed is None else seed}
        save_json(output / "generation_request.json", request)
        if reference_metadata:
            save_json(output / "reference_metadata.json", reference_metadata)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=self.config["gpus"],
                   DIFFSYNTH_ATTENTION_IMPLEMENTATION="torch",
                   PYTORCH_ALLOC_CONF="expandable_segments:True")
        deploy = Path(self.config["deploy"])
        env["PYTHONPATH"] = ":".join((str(deploy / "quant-deps"), str(deploy / "optional_flash"),
                                     str(deploy), self.config["repo"]))
        started = time.monotonic()
        command = [self.config["python"], "-u", str(Path(__file__).with_name("worker.py")),
                   str(output / "generation_request.json")]
        error = None
        returncode = None
        try:
            with (output / "generation.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=self.config["timeout"])
                returncode = result.returncode
        except subprocess.TimeoutExpired:
            error = "H3 subprocess timed out and was terminated"
        save_json(output / "generation_metadata.json",
                  {"command": command, "seconds": time.monotonic()-started,
                   "returncode": returncode, "error": error,
                   "first_frame_semantics": "image reference, not locked",
                   "depth_conditioned": False, "semantic_conditioned": False})
        if error or returncode or not (output / "generated.mp4").exists():
            raise RuntimeError(f"H3 failed; see {output / 'generation.log'}")
        return output / "generated.mp4"
