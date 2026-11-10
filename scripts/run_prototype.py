import argparse
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.decision import decide, plan_rollout
from common import load_config, save_json
from compiler.compile import compile_world
from matching.matcher import match
from memory.observation_store import ObservationStore
from perception.detector import PerceptionPipeline
from perception.frame_sampler import sample_frames
from perception.visualize import draw_frame
from perception.quality import reference_assessability
from perception.embeddings import CropEncoder
from PIL import Image
from renderer.minimax_h3 import H3Renderer
from state.db import WorldDB
from state.models import WorldObject
from state.adapters import inverse_scene
from scripts.test_h3 import scene
from verifier.discrepancy import discrepancy
from verifier.reconcile import apply_reconciliation


def run(config, output, steps=None, replay_video=None, max_reroll=None, inverse_path=None,
        whitebox_source=None, timestamp=0.0):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new episode output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    episode_id = output.name
    initial, camera = scene()
    if inverse_path:
        if not whitebox_source:
            raise ValueError("--inverse-scene requires --whitebox-source for the existing renderer")
        initial, camera = inverse_scene(inverse_path, timestamp, config["renderer"]["width"], config["renderer"]["height"])
        camera["whitebox_source_root"] = str(Path(whitebox_source).resolve())
    world = WorldDB(config["world_db"])
    if inverse_path and world.query_world():
        raise ValueError("Import into a new working DB; do not replace the existing world implicitly")
    if not world.query_world():
        world.set_meta("camera", camera)
        world.set_meta("tick", 0)
        for obj in initial:
            world.save(obj, "prototype initial state")
    camera = world.get_meta("camera")
    if camera is None:
        raise ValueError("Compiler needs scene camera; matching alone supports missing camera")
    save_json(output / "world_before.json", world.snapshot())
    store = ObservationStore(config["observation_db"])
    history = store.get_recent_decisions()
    plan = plan_rollout(config["agent"], world, "Show a short stable rollout of the current world.", history)
    save_json(output / "agent_plan.json", plan)
    current_intention = plan["rollout_intention"]
    decisions = []
    retry_limit = config["agent"]["max_reroll"] if max_reroll is None else max_reroll
    if retry_limit < 0:
        raise ValueError("max_reroll must be nonnegative")
    for attempt in range(retry_limit + 1):
        objects = world.query_world()
        attempt_dir = output / f"attempt_{attempt:02d}"
        compiled = compile_world(objects, camera, attempt_dir / "proxy", current_intention)
        if any(o.appearance_embedding is None for o in objects if o.state.get("matchable", True)):
            encoder = CropEncoder(device="cpu")
            reference = Image.open(compiled["first_frame"]).convert("RGB")
            for obj in objects:
                projection = compiled["projections"].get(obj.object_id)
                if obj.appearance_embedding is None and projection and projection.get("bbox") and obj.state.get("matchable", True):
                    feature = encoder.encode_object_crop(reference, None, projection["bbox"])
                    if feature is not None:
                        obj.appearance_embedding = feature.tolist()
                        world.save(obj, "DINOv2 appearance metadata from authoritative proxy, not generated video")
            encoder.close()
        if replay_video:
            video = Path(replay_video).resolve()
            if not video.exists():
                raise FileNotFoundError(video)
            save_json(attempt_dir / "replay_metadata.json",
                      {"video": str(video), "renderer_called_this_attempt": False})
        else:
            renderer_config = dict(config["renderer"])
            if steps:
                renderer_config["steps"] = steps
            video = H3Renderer(renderer_config).render(
                compiled["prompt"], compiled["first_frame"], output_dir=attempt_dir,
                seed=renderer_config["seed"]+attempt)
        frames = sample_frames(video, attempt_dir / "sampled_frames",
                               config["perception"]["frame_interval_sec"])
        reference_frame = {"path": compiled["first_frame"], "frame_id": 0, "timestamp": 0.0}
        reference_observations = PerceptionPipeline(config["perception"]).detect_frame(
            reference_frame, attempt_dir / "reference_perception", f"{episode_id}:reference:{attempt}",
            source="COMPILED_REFERENCE")
        reference_match = match(objects, reference_observations, camera, config["matching"]).to_dict()
        quality = reference_assessability(reference_match)
        save_json(attempt_dir / "reference_assessability.json",
                  {"quality": quality, "matching": reference_match,
                   "observations": [o.to_dict() for o in reference_observations]})
        detector = PerceptionPipeline(config["perception"])
        frame_results = []
        observation_frames = []
        track_world = {}
        for frame in frames:
            observations = detector.detect_frame(frame, attempt_dir / "segmentation" / str(frame["frame_id"]),
                                                  f"{episode_id}:{attempt}")
            for item in observations:
                store.add_object(episode_id, attempt, item)
            store.add_frame(episode_id, attempt, frame)
            result = match(objects, observations, camera, config["matching"], track_world)
            frame_results.append({"frame_id": frame["frame_id"], "matching": result.to_dict()})
            observation_frames.append({"frame": frame, "objects": [o.to_dict() for o in observations]})
            draw_frame(frame, observations, result.to_dict(), objects, camera,
                       attempt_dir / "sampled_frames" / f"debug_{frame['frame_id']:06d}.png")
        report = discrepancy(f"{episode_id}_{attempt:02d}", frame_results, config["matching"], quality)
        store.add_report(episode_id, attempt, report)
        save_json(attempt_dir / "observations.json", observation_frames)
        save_json(attempt_dir / "matching.json", frame_results)
        save_json(attempt_dir / "discrepancy.json", report)
        decision = decide(config["agent"], world, current_intention, report, (history+decisions)[-3:], store, frames)
        try:
            applied = apply_reconciliation(world, decision)
            decision["applied_edits"] = applied
        except ValueError as error:
            decision["rejected_reconciliation"] = str(error)
            decision["action"] = "REGENERATE"
        store.add_decision(episode_id, attempt, decision)
        save_json(attempt_dir / "agent_decision.json", decision)
        decisions.append(decision)
        if decision["action"] != "REGENERATE":
            if decision["action"] == "ACCEPT":
                world.set_meta("last_accepted_rollout", report["rollout_id"])
                world.set_meta("tick", world.get_meta("tick", 0)+1)
            break
        if replay_video:
            decision["retry_exhausted"] = "Replay mode cannot generate a new video"
            save_json(attempt_dir / "agent_decision.json", decision)
            break
        current_intention += " Correct the previous rollout's missing or inconsistent objects."
    save_json(output / "status.json", {"accepted": decisions[-1]["action"] == "ACCEPT",
                                     "action": decisions[-1]["action"], "attempts": len(decisions),
                                     "retry_exhausted": decisions[-1]["action"] == "REGENERATE"})
    save_json(output / "world_after.json", world.snapshot())
    save_json(output / "agent_decisions.json", decisions)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--replay-video")
    parser.add_argument("--max-reroll", type=int)
    parser.add_argument("--inverse-scene")
    parser.add_argument("--whitebox-source")
    parser.add_argument("--timestamp", type=float, default=0.0)
    parser.add_argument("--world-db")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.world_db:
        config["world_db"] = args.world_db
    output = args.output or str(Path(config["output_root"]) / ("episode_" + uuid.uuid4().hex[:10]))
    try:
        print(run(config, output, args.steps, args.replay_video, args.max_reroll,
                  args.inverse_scene, args.whitebox_source, args.timestamp))
    except Exception as error:
        if not isinstance(error, FileExistsError):
            save_json(Path(output) / "error.json",
                      {"type": type(error).__name__, "message": str(error), "accepted": False})
        raise
