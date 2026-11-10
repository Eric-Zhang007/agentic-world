"""Launch the fixed six-case experiment in the existing isolated Comfy env."""
import argparse
import os
import json
from pathlib import Path
import shutil
import subprocess
import time


def main(root, gpus):
    project = Path(__file__).resolve().parents[1]
    root = root.resolve()
    command = ["/home/star/agentic-world-comfy/.venv/bin/python", "-u",
               str(project / "scripts/run_comfy_control.py"),
               "--comfy", "/home/star/agentic-world-comfy/ComfyUI",
               "--reference-image", str(project / "outputs/h3_scene_20261008/frame_011.png"),
               "--control-video", str(root / "control.npz"), "--cases", str(root / "cases.json"),
               "--output", str(root / "render"), "--width", "1024", "--height", "768",
               "--frames", "73", "--steps", "40", "--seed", "43",
               "--kernel-backend", "triton", "--anchor-first", "--prompt", (root / "prompt.txt").read_text()]
    cases=json.loads((root/"cases.json").read_text())
    devices=gpus.split(",")
    if len(set(devices))!=len(devices) or not 1<=len(devices)<=3:
        raise ValueError("Use one to three distinct explicitly assigned GPUs")
    output=root/"render"
    output.mkdir(exist_ok=False)
    jobs=[];published={};worker_metadata={}
    try:
        for index,gpu in enumerate(devices):
            folder=root/f"worker_{index}";folder.mkdir(exist_ok=False)
            subset=cases[index::len(devices)]
            case_path=folder/"cases.json";case_path.write_text(json.dumps(subset,indent=2))
            worker_command=command.copy()
            worker_command[worker_command.index("--cases")+1]=str(case_path)
            worker_command[worker_command.index("--output")+1]=str(folder/"render")
            log=(folder/"generation.log").open("w")
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,HF_ENDPOINT="https://hf-mirror.com")
            process=subprocess.Popen(worker_command,stdout=log,stderr=subprocess.STDOUT,env=environment)
            jobs.append((process,log,folder,gpu))
        while True:
            for process,log,folder,gpu in jobs:
                try:metadata=json.loads((folder/"render/metadata.json").read_text())
                except (FileNotFoundError,json.JSONDecodeError):continue
                worker_metadata[gpu]={k:v for k,v in metadata.items() if k!="runs"}
                for run in metadata["runs"]:
                    label=run["label"]
                    if label in published:continue
                    for artifact in (folder/"render").glob(label+"*"):
                        shutil.copy2(artifact,output/artifact.name)
                    published[label]={**run,"gpu":gpu}
                    print(json.dumps({"completed":label,"gpu":gpu,"sample_seconds":run["sample_seconds"]}),flush=True)
            (output/"metadata.json").write_text(json.dumps({"runs":list(published.values()),"workers":worker_metadata},indent=2))
            if any(process.poll() not in (None,0) for process,_,_,_ in jobs):
                raise RuntimeError("H3 worker failed; inspect worker generation logs")
            if all(process.poll() is not None for process,_,_,_ in jobs):break
            time.sleep(5)
        if len(published)!=len(cases):raise RuntimeError("Incomplete generation matrix")
    finally:
        for process,log,_,_ in jobs:
            if process.poll() is None:process.terminate()
        for process,log,_,_ in jobs:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--gpus", default="3")
    a = p.parse_args()
    main(a.root, a.gpus)
