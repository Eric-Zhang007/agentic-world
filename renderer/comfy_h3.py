"""Isolated native ComfyUI renderer; leaves the working DiffSynth backend untouched."""
import json
import os
from pathlib import Path
import subprocess
import time


class ComfyH3Renderer:
    def __init__(self, python="/home/star/agentic-world-comfy/.venv/bin/python",
                 repo="/home/star/agentic-world-comfy/ComfyUI", gpu="3",
                 steps=40, frames=73, width=832, height=480, timeout=3600, kernel_backend="default",
                 lora="/home/star/agentic-world-comfy/ComfyUI/models/loras/epoch-15.safetensors"):
        self.python, self.repo, self.gpu = python, repo, gpu
        self.steps, self.frames = steps, frames
        self.width, self.height, self.timeout = width, height, timeout
        self.kernel_backend = kernel_backend
        self.lora = lora

    def render(self, prompt, source_video, output_dir, first_frame=None,
               control_video=None, seed=43, hud_profile=None, strength=1.0,
               control_end=1.0, cache_threshold=0.0, controls=None, anchor_first=False,last_frame=None,
               prefix_video=None, appearance_images=(), reference_video=None, history_pair=None):
        """Native video references are separate from RGB ControlNet and prefix guides.

        Omitted controls default to disabled. Depth remains an explicit opt-in.
        """
        if controls is None:
            if control_video is not None:
                raise ValueError('Specify controls explicitly when providing a control video')
            controls = []
        if history_pair is not None:
            if not first_frame or not reference_video or appearance_images:
                raise ValueError('Paired memory needs first frame and native whitebox video, without legacy crops')
            from renderer.memory_conditioning import memory_prompt
            prompt = memory_prompt(prompt, (self.frames-1)/24, True)
        elif reference_video and '<Picture' not in prompt and '<Video' not in prompt:
            from renderer.memory_conditioning import memory_prompt
            prompt = memory_prompt(prompt, (self.frames-1)/24, False)
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        if any((out / name).exists() for name in ("controlled.mp4", "baseline.mp4")):
            raise FileExistsError("Choose a new output directory; a rollout already exists here")
        video = out / ("baseline.mp4" if controls is None and strength == 0 else "controlled.mp4")
        worker = Path(__file__).resolve().parents[1] / "scripts/run_comfy_control.py"
        command = [
            self.python, "-u", str(worker), "--comfy", self.repo,
            "--output", str(out),
            "--prompt", prompt, "--steps", str(self.steps), "--frames", str(self.frames),
            "--width", str(self.width), "--height", str(self.height), "--seed", str(seed)]
        command.extend(("--kernel-backend",self.kernel_backend))
        if self.lora:
            command.extend(('--lora', str(self.lora)))
        if history_pair:
            command.extend(('--history-pair', str(Path(history_pair).resolve())))
        if len(appearance_images)>8:
            raise ValueError('H3 supports at most nine total reference images')
        for image in appearance_images:
            path = Path(image).resolve()
            if not path.is_file():raise FileNotFoundError(path)
            command.extend(('--appearance-image',str(path)))
        if source_video is not None:
            command.extend(("--source",str(Path(source_video).resolve())))
        if anchor_first:
            command.append("--anchor-first")
        command.extend(("--strength",str(strength),"--control-end",str(control_end),
                        "--cache-threshold",str(cache_threshold)))
        if controls is not None:
            cases=out/"controls_request.json"
            cases.write_text(json.dumps([{"label":"controlled","controls":controls,
                                          "cache_threshold":cache_threshold}],indent=2))
            command.extend(("--cases",str(cases)))
        for flag, path in (("--reference-image", first_frame), ("--control-video", control_video),
                           ("--hud-profile", hud_profile),("--last-frame",last_frame),
                           ("--prefix-video", prefix_video), ("--reference-video", reference_video)):
            if path is not None:
                command.extend((flag, str(Path(path).resolve())))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu, HF_ENDPOINT="https://hf-mirror.com")
        started = time.monotonic()
        record = {"command": command, "gpu": self.gpu,
                  "control": ("disabled" if controls == [] else
                              "explicit controls" if controls is not None else
                              "external RGB" if control_video else "source-derived Canny"),
                  "controls": controls,
                  "native_reference_video": str(Path(reference_video).resolve()) if reference_video else None}
        try:
            with (out / "generation.log").open("w") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=self.timeout, check=False)
            record["returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"ComfyUI renderer failed; see {out / 'generation.log'}")
            if not video.exists() or not video.stat().st_size:
                raise RuntimeError("ComfyUI exited without a video")
            record["video_path"] = str(video)
            return video
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            (out / "generation_metadata.json").write_text(json.dumps(record, indent=2))
