"""Render whitebox JSON using standalone Godot and a private Linux X display."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def vector(value, name, length=3):
    if not isinstance(value, list) or len(value) != length or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in value
    ):
        raise ValueError(f"{name} must contain {length} finite numbers")


def camera(value):
    vector(value["position"], "camera.position")
    vector(value["target"], "camera.target")
    if value["position"] == value["target"]:
        raise ValueError("Camera position and target must differ")
    if 'intrinsics' in value:
        k=value['intrinsics']
        for key in ('width','height','fx','fy'):
            if not isinstance(k.get(key),(int,float)) or not math.isfinite(k[key]) or k[key]<=0:
                raise ValueError('Invalid camera intrinsics: '+key)
        if abs(k['cx']-k['width']/2)>1e-6 or abs(k['cy']-k['height']/2)>1e-6:
            raise ValueError('This renderer currently requires a centered principal point')
    if 'up' in value:vector(value['up'],'camera.up')


def validate(scene, timeline):
    if scene.get("schema_version") != 1:
        raise ValueError("scene.schema_version must be 1")
    terrain = scene["terrain"]
    if type(terrain.get("seed", scene.get("seed"))) is not int:
        raise ValueError("terrain.seed must be an integer")
    vector(terrain["size"], "terrain.size", 2)
    if min(terrain["size"]) <= 0:
        raise ValueError("Terrain dimensions must be positive")
    resolution = terrain["resolution"]
    if type(resolution) is not int or not 1 <= resolution <= 512:
        raise ValueError("terrain.resolution must be an integer in [1, 512]")
    for key in ("height_amplitude", "flatten_radius"):
        if key in terrain and (not math.isfinite(terrain[key]) or terrain[key] < 0):
            raise ValueError(f"terrain.{key} must be finite and nonnegative")
    camera(scene["camera"])
    ids = set()
    for entity in scene["entities"]:
        eid = str(entity["id"])
        if eid in ids:
            raise ValueError(f"Duplicate entity id: {eid}")
        ids.add(eid)
        if entity["kind"] not in ("actor", "box", "staff", "stairs"):
            raise ValueError(f"Unknown entity kind: {entity['kind']}")
        vector(entity["position"], f"{eid}.position")
        vector(entity["size"], f"{eid}.size")
        if min(entity["size"]) <= 0:
            raise ValueError(f"{eid}.size must be positive")
        if "color" in entity:
            vector(entity["color"], f"{eid}.color")
            if any(c < 0 or c > 1 for c in entity["color"]):
                raise ValueError(f"{eid}.color components must be in [0, 1]")
        if entity["kind"] == "stairs":
            steps = entity.get("steps", entity.get("step_count", 6))
            if type(steps) is not int or not 1 <= steps <= 256:
                raise ValueError("stairs.steps must be an integer in [1, 256]")
        validate_update(entity)
    if timeline is None:
        return
    fps = timeline["fps"]
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("timeline.fps must be positive and finite")
    previous = -1.0
    for frame in timeline["frames"]:
        time = frame["time"]
        if not math.isfinite(time) or time < 0 or time <= previous:
            raise ValueError("Frame times must be nonnegative and strictly increasing")
        previous = time
        seen = set()
        for entity in frame.get("entities", []):
            eid = str(entity["id"])
            if eid not in ids or eid in seen:
                raise ValueError(f"Unknown or duplicate timeline entity: {eid}")
            seen.add(eid)
            validate_update(entity)
        if "camera" in frame:
            camera(frame["camera"])


def validate_update(entity):
    if "position" in entity:
        vector(entity["position"], "entity.position")
    if "rotation_y" in entity and not math.isfinite(entity["rotation_y"]):
        raise ValueError("rotation_y must be finite")
    for key, point in entity.get("pose", {}).items():
        vector(point, f"pose.{key}")
    if "weapon" in entity:
        vector(entity["weapon"]["a"], "weapon.a")
        vector(entity["weapon"]["b"], "weapon.b")


@contextlib.contextmanager
def display(width, height, use_display):
    env = os.environ.copy()
    if use_display or sys.platform != "linux":
        yield env
        return
    xvfb = shutil.which("Xvfb") or str(HERE / "runtime/x11/usr/bin/Xvfb")
    if not Path(xvfb).is_file():
        raise RuntimeError("Xvfb missing. Run bootstrap_server.sh or use --use-display.")
    libs = HERE / "runtime/x11/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = str(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["LP_NUM_THREADS"] = "4"
    read_fd, write_fd = os.pipe()
    process = None
    try:
        # Xvfb chooses an unused display atomically; no shared desktop is touched.
        process = subprocess.Popen(
            [xvfb, "-displayfd", str(write_fd), "-screen", "0", f"{width}x{height}x24",
             "-nolisten", "tcp", "-ac", "-noreset"], env=env, pass_fds=(write_fd,),
            stdout=subprocess.DEVNULL,
        )
        os.close(write_fd)
        write_fd = None
        if not select.select([read_fd], [], [], 20)[0]:
            raise RuntimeError("Xvfb did not start within 20 seconds")
        number = os.read(read_fd, 128).decode().strip()
        if not number.isdigit():
            raise RuntimeError("Xvfb failed to allocate a display")
        env["DISPLAY"] = ":" + number
        yield env
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", default="selected", help="selected (nine evenly spaced), all, or comma-separated indices")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--render-pass", choices=("preview","depth"), default="preview")
    parser.add_argument("--godot", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--use-display", action="store_true")
    parser.add_argument("--mp4", action="store_true", help="Render every frame and encode timeline.mp4")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if min(args.width, args.height) < 64 or max(args.width, args.height) > 4096:
        parser.error("Image dimensions must be in [64, 4096]")
    scene = json.loads(args.scene.read_text(encoding="utf-8-sig"))
    timeline = json.loads(args.timeline.read_text(encoding="utf-8-sig")) if args.timeline else None
    validate(scene, timeline)
    if args.mp4:
        if not timeline or not timeline["frames"]:
            parser.error("--mp4 requires a nonempty timeline")
        if not shutil.which("ffmpeg"):
            parser.error("--mp4 requires ffmpeg")
        if args.width % 2 or args.height % 2:
            parser.error("--mp4 requires even image dimensions")
        times = [f["time"] for f in timeline["frames"]]
        if any(abs((b-a) - 1/timeline["fps"]) > 1e-5 for a, b in zip(times, times[1:])):
            parser.error("--mp4 requires uniform frame times spaced at 1/fps")
        args.frames = "all"
    if args.frames not in ("all", "selected"):
        if timeline is None:
            parser.error("Frame indices require --timeline")
        indices = [int(token) for token in args.frames.split(",")]
        if any(i < 0 or i >= len(timeline["frames"]) for i in indices):
            parser.error("Frame index out of range")
    godot = args.godot or shutil.which("godot4") or shutil.which("godot")
    godot = godot or str(HERE / "runtime/Godot_v4.5-stable_linux.x86_64")
    if not Path(godot).is_file():
        parser.error("Godot missing. Run bootstrap_server.sh or pass --godot /path/to/godot.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Refuse stale results rather than accidentally validating a previous run.
    if any(output.glob("*.png")) or (output / "manifest.json").exists():
        parser.error("Output already contains renders; choose a fresh directory")
    command = [godot, "--path", str(HERE), "--rendering-method", "gl_compatibility",
               "--audio-driver", "Dummy", "--resolution", f"{args.width}x{args.height}",
               "--script", str(HERE / "render.gd")]
    if sys.platform == "linux":
        command += ["--display-driver", "x11"]
    command += ["--", "--scene", str(args.scene.resolve()), "--output", str(output),
                "--width", str(args.width), "--height", str(args.height), "--frames", args.frames,
                "--render-pass", args.render_pass]
    if args.timeline:
        command += ["--timeline", str(args.timeline.resolve())]
    with display(args.width, args.height, args.use_display) as env:
        subprocess.run(command, env=env, check=True, timeout=args.timeout)
    manifest = json.loads((output / "manifest.json").read_text())
    if len(manifest["images"]) < (4 if args.render_pass == "preview" else 1):
        raise RuntimeError("Missing required review views")
    if args.mp4:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-framerate",
                        str(timeline["fps"]), "-i", str(output / "frame_%06d.png"),
                        "-frames:v", str(len(timeline["frames"])), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "timeline.mp4")],
                       check=True, timeout=args.timeout)
    print(json.dumps({"output": str(output), "images": len(manifest["images"]),
                      "renderer": manifest["renderer"], "mp4": args.mp4,
                      "png_paths": [str(output / item["file"]) for item in manifest["images"]],
                      "video_path": str(output / "timeline.mp4") if args.mp4 else None}))


if __name__ == "__main__":
    main()
