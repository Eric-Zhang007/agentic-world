"""Image + user action -> persistent coarse world -> multi-clip verified action."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.action_planner import construct_and_plan,assess_phase,refine_phase
from compiler.world_controls import compile_timeline
from matching.matcher import match
from memory.observation_store import ObservationStore
from perception.detector import PerceptionPipeline
from perception.embeddings import CropEncoder
from perception.frame_sampler import sample_frames
from perception.visualize import draw_frame
from renderer.comfy_h3 import ComfyH3Renderer
from state.action_session import validate_plan,initialize_world,start_action,propose_timeline,apply_assessment
from state.db import WorldDB
from state.models import WorldObject
from verifier.discrepancy import discrepancy


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load(a.config.read_text())
    world=WorldDB(out/"world.sqlite");store=ObservationStore(out/"observations.sqlite")
    if a.cancel:
        action=world.get_meta("active_action")
        if not action:raise ValueError("No action to cancel")
        action["status"]="CANCELLED";action["user_input_ready"]=True
        world.set_meta("active_action",action);save(out/"action_state.json",action)
        print(json.dumps({"status":"CANCELLED"}));return
    if a.resume:
        action=world.get_meta("active_action")
        if not action or action["status"]!="RUNNING":raise ValueError("No running action to resume")
        image=Path(world.get_meta("initial_image"));goal=action["goal"]
        settings=world.get_meta("renderer_settings")
        a.width,a.height,a.frames=settings["width"],settings["height"],settings["frames"]
    elif world.query_world():
        current=world.get_meta("active_action")
        if current and current["status"] not in {"COMPLETE","CANCELLED"}:
            raise ValueError("Action is unfinished; use --resume or --cancel before a new goal")
        if not a.goal:raise ValueError("A new action needs --goal")
        settings=world.get_meta("renderer_settings")
        a.width,a.height,a.frames=settings["width"],settings["height"],settings["frames"]
        image=Path(world.get_meta("initial_image"));goal=a.goal
        summary=world.snapshot()
        for obj in summary["objects"]:obj.pop("appearance_embedding",None)
        plan=construct_and_plan(config["agent"],world.get_meta("continuity_image"),goal,a.frames/24,a.max_phases,summary)
        validate_plan(plan,a.frames/24,a.max_phases)
        if {o["object_id"] for o in plan["objects"]}!={o.object_id for o in world.query_world()}:
            raise ValueError("New action changed persistent identity set")
        serial=world.get_meta("action_serial",0)+1;world.set_meta("action_serial",serial)
        if current:save(out/f"action_{serial-1:03d}_state.json",current)
        save(out/f"action_{serial:03d}_plan.json",plan)
        action=start_action(world,goal,plan)
    else:
        if not a.image or not a.goal:raise ValueError("New action needs --image and --goal")
        if world.query_world():raise ValueError("Output already contains a world; use --resume")
        image=out/"initial.png";Image.open(a.image).convert("RGB").resize((a.width,a.height)).save(image)
        goal=a.goal
        print(json.dumps({"stage":"construct_and_plan","goal":goal}),flush=True)
        plan=construct_and_plan(config["agent"],image,goal,a.frames/24,a.max_phases)
        save(out/"initial_plan.json",plan)
        validate_plan(plan,a.frames/24,a.max_phases)
        initialize_world(world,plan,image,a.width,a.height)
        encoder=CropEncoder("cpu")
        initial=Image.open(image)
        for obj in world.query_world():
            box=obj.attributes["initial_bbox_normalized"]
            vector=encoder.encode_object_crop(initial,None,[v*(a.width if i%2==0 else a.height) for i,v in enumerate(box)])
            obj.appearance_embedding=vector.tolist() if vector is not None else None
            world.save(obj,"Original-image DINOv2 appearance embedding")
        encoder.close()
        world.set_meta("initial_image",str(image));world.set_meta("continuity_image",str(image))
        world.set_meta("renderer_settings",{"width":a.width,"height":a.height,"frames":a.frames})
        action=start_action(world,goal,plan)
        save(out/"world_before.json",world.snapshot())
    save(out/"action_state.json",action)
    save(out/"run_summary.json",{"status":action["status"],"user_input_ready":action["user_input_ready"],
                                "attempts":action["attempts"],"execution":"running"})
    renderer=ComfyH3Renderer(gpu=a.gpu,steps=a.steps,frames=a.frames,width=a.width,height=a.height,
                            kernel_backend=a.kernel_backend)
    perception_config={**config["perception"],"device":a.perception_device,
                       "categories":sorted({o.category for o in world.query_world()}|{"person","tree","dog","vehicle"})}
    perception_config.pop("hud_profile",None)
    pipeline=PerceptionPipeline(perception_config)
    for _ in range(a.max_rollouts):
        action=world.get_meta("active_action")
        if action["status"]=="COMPLETE":break
        attempt=action["attempts"];phase_index=action["phase_index"]
        phase=deepcopy(action["plan"]["phases"][phase_index])
        last=action["history"][-1] if action["history"] else None
        if last and last["phase_index"]==phase_index:
            refined=refine_phase(config["agent"],phase,last["decision"],world,world.get_meta("continuity_image"))
            for key in ("name","duration_sec","completion_criterion"):
                if refined.get(key)!=phase[key]:raise ValueError(f"Refinement changed phase contract: {key}")
            candidate=deepcopy(action["plan"]);candidate["phases"][phase_index]=refined
            validate_plan(candidate,a.frames/24,max(a.max_phases,len(candidate["phases"])))
            phase=refined
        serial=world.get_meta("action_serial",0)
        segment=out/(f"action_{serial:03d}_segment_{attempt:03d}" if serial else f"segment_{attempt:03d}")
        observation_attempt=serial*100000+attempt
        if segment.exists():raise FileExistsError(f"Incomplete previous attempt at {segment}; inspect before rerun")
        segment.mkdir()
        save(segment/"world_before.json",world.snapshot());save(segment/"phase.json",phase)
        phase_frames=round(phase["duration_sec"]*24)
        if phase_frames%17!=5:raise ValueError("Invalid phase duration frame grid")
        renderer.frames=phase_frames
        timeline=propose_timeline(world,phase,phase_frames)
        manifest=compile_timeline(timeline,segment/"controls",phase_frames)
        save(segment/"candidate_timeline.json",timeline)
        prompt=(action["plan"]["scene_description"]+" Match the actor appearances in <Picture 1>. "+
                phase["render_prompt"]+" Continuous coherent motion, no interface overlays. "+
                "Fine natural limb motion and environmental details are free; preserve actor identity.")
        print(json.dumps({"stage":"render","attempt":attempt,"phase":phase_index,"name":phase["name"]}),flush=True)
        video=renderer.render(prompt,None,segment/"render",first_frame=world.get_meta("continuity_image"),
            seed=a.seed+attempt,controls=[{"path":str(segment/"controls/layout.npz"),"strength":a.strength,"end":.65}],
            cache_threshold=a.cache_threshold,anchor_first=True)
        frames=sample_frames(video,segment/"sampled_frames",5.0)
        rows=[];observed=[];track_world={}
        for frame in frames:
            observations=pipeline.detect_frame(frame,segment/"segmentation"/str(frame["frame_id"]),f"{out.name}:{segment.name}")
            idx=min(phase_frames-1,round(frame["timestamp"]*24));expected=manifest["expected"][idx]
            objects=[WorldObject(**o) for o in expected["objects"]]
            result=match(objects,observations,expected["camera"],config["matching"],track_world).to_dict()
            rows.append({"frame_id":frame["frame_id"],"matching":result})
            observed.extend(o.to_dict() for o in observations)
            store.add_frame(out.name,observation_attempt,frame)
            for obj in observations:store.add_object(out.name,observation_attempt,obj)
            draw_frame(frame,observations,result,objects,expected["camera"],segment/f"debug_{idx:03d}.png")
        report=discrepancy(segment.name,rows,config["matching"])
        report["world_scope"]="partial and single-image inferred; unmatched background is not proof of hallucination"
        report["geometry_scope"]="coarse intended trajectory, not measured future geometry"
        save(segment/"observations.json",observed);save(segment/"matching.json",rows)
        save(segment/"discrepancy.json",report);store.add_report(out.name,observation_attempt,report)
        print(json.dumps({"stage":"assess","attempt":attempt,"summary":report["summary"]}),flush=True)
        review_frames=sample_frames(video,segment/"review_frames",phase_frames/24/8)
        completed_phases=[{"phase_index":h["phase_index"],"name":action["plan"]["phases"][h["phase_index"]]["name"],
                           "rollout_id":h["rollout_id"],"evidence":h["decision"]["completion_evidence"]}
                          for h in action["history"] if h["decision"].get("phase_complete")]
        decision=assess_phase(config["agent"],goal,phase,report,review_frames,image,
            phase_index==len(action["plan"]["phases"])-1,completed_phases,world.get_meta("continuity_image"))
        save(segment/"agent_decision_raw.json",decision)
        action=apply_assessment(world,decision,timeline,segment.name)
        effective=action["history"][-1]["decision"]
        if effective["action"]=="ACCEPT" and effective["phase_complete"]:
            world.set_meta("continuity_image",frames[-1]["path"])
        store.add_decision(out.name,observation_attempt,effective)
        save(segment/"agent_decision.json",effective);save(segment/"world_after.json",world.snapshot())
        save(out/"action_state.json",action);save(out/"world_after.json",world.snapshot())
        print(json.dumps({"stage":"decision","attempt":attempt,"decision":effective,"status":action["status"]}),flush=True)
    action=world.get_meta("active_action")
    save(out/"action_state.json",action)
    save(out/"run_summary.json",{"status":action["status"],"user_input_ready":action["user_input_ready"],
        "attempts":action["attempts"],"accepted_segments":action["accepted_segments"],
        "budget_exhausted":action["status"]!="COMPLETE","world_db":str(out/"world.sqlite"),
        "limitation":"Completion judged by sparse VLM frames; intended geometry is not measured dynamics"})
    print(json.dumps({"stage":"finished","status":action["status"],"user_input_ready":action["user_input_ready"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path("configs/default.yaml"))
    p.add_argument("--image",type=Path);p.add_argument("--goal");p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",action="store_true");p.add_argument("--max-phases",type=int,default=3)
    p.add_argument("--cancel",action="store_true")
    p.add_argument("--max-rollouts",type=int,default=3);p.add_argument("--steps",type=int,default=20)
    p.add_argument("--frames",type=int,default=73);p.add_argument("--width",type=int,default=832)
    p.add_argument("--height",type=int,default=480);p.add_argument("--seed",type=int,default=81)
    p.add_argument("--gpu",default="1");p.add_argument("--perception-device",default="cuda:0")
    p.add_argument("--strength",type=float,default=.35);p.add_argument("--cache-threshold",type=float,default=0)
    p.add_argument("--kernel-backend",choices=("default","triton"),default="default")
    main(p.parse_args())
