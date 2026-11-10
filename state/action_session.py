"""Durable action lifecycle. Generated-only facts never authorize object creation."""
from copy import deepcopy
import math

import numpy as np
from state.models import WorldObject


def validate_plan(plan,segment_seconds,max_segments):
    objects=plan.get("objects",[]);phases=plan.get("phases",[])
    if not 1<=len(objects)<=4 or not 1<=len(phases)<=max_segments:
        raise ValueError("Plan needs 1-4 objects and bounded phases")
    ids=[o["object_id"] for o in objects]
    if len(set(ids))!=len(ids):raise ValueError("Duplicate object IDs")
    for relation in plan.get("relations",[]):
        if relation["predicate"]!="held_by" or relation["subject"] not in ids or relation["object"] not in ids or relation["subject"]==relation["object"]:
            raise ValueError("Invalid held-object relation")
    holders={r["subject"]:r["object"] for r in plan.get("relations",[])}
    if len(holders)!=len(plan.get("relations",[])):raise ValueError("Multiple holders for one object")
    for child in holders:
        seen=set();node=child
        while node in holders:
            if node in seen:raise ValueError("Cyclic held-object relation")
            seen.add(node);node=holders[node]
    for obj in objects:
        b=obj["bbox_normalized"]
        if len(b)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in b) or b[0]>=b[2] or b[1]>=b[3]:
            raise ValueError("Invalid observed box")
        if not .2<=obj["approx_height"]<=5 or not 0<=obj["confidence"]<=1:
            raise ValueError("Unbounded inferred size/confidence")
        if obj.get("control_level","layout") not in {"layout","identity_only"}:
            raise ValueError("Invalid control granularity")
    for phase in phases:
        if not phase.get("completion_criterion") or not phase.get("render_prompt"):
            raise ValueError("Phase lacks prompt or completion criterion")
        if not math.isfinite(phase["duration_sec"]) or min(abs(phase["duration_sec"]-x) for x in (segment_seconds,73/24,124/24))>.02:
            raise ValueError("Phase duration does not match renderer horizon")
        if not .3<=phase.get("motion_end_fraction",.65)<=.9:
            raise ValueError("Motion must leave a visible settling interval")
        moving=[]
        for move in phase["moves"]:
            if move["object_id"] not in ids:raise ValueError("Move references unknown object")
            moving.append(move["object_id"])
            delta=np.asarray(move["delta_world"],float)
            if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>8:
                raise ValueError("Implausible coarse displacement")
        if len(set(moving))!=len(moving):raise ValueError("Duplicate object moves")
        delta=np.asarray(phase["camera_delta"],float)
        if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>5:
            raise ValueError("Camera move exceeds single-image support")
        if phase.get("camera_target_id") not in ids+[None]:raise ValueError("Unknown camera target")
        if not set(phase.get("controlled_object_ids",ids)).issubset(ids):raise ValueError("Unknown controlled object")


def initialize_world(world,plan,image,width,height):
    if world.query_world():raise ValueError("Image initialization requires an empty working world")
    f=1.1*width
    camera={"width":width,"height":height,"intrinsics":[[f,0,width/2],[0,f,height/2],[0,0,1]],
            "world_to_camera":[[1,0,0,0],[0,0,-1,1.5],[0,1,0,0],[0,0,0,1]],
            "position":[0,0,1.5],"target":[0,5,1.5],"provenance":"INFERRED_single_image"}
    for obj in plan["objects"]:
        x0,y0,x1,y1=obj["bbox_normalized"]
        z=f*obj["approx_height"]/((y1-y0)*height)
        x=((x0+x1)*width/2-width/2)*z/f
        up=1.5-((y0+y1)*height/2-height/2)*z/f
        size=[(x1-x0)*width*z/f,.45*obj["approx_height"],obj["approx_height"]]
        evidence_id="initial:"+obj["object_id"]
        world.add_evidence(evidence_id,"ORIGINAL",obj["object_id"],
                           {"image":str(image),"bbox_normalized":obj["bbox_normalized"],"interpretation":"GPT image proposal, not measured 3D"})
        world.save(WorldObject(object_id=obj["object_id"],category=obj["category"],position=[x,z,up],
            bbox_3d=size,attributes={"description":obj["description"],"initial_bbox_normalized":obj["bbox_normalized"],
                "geometry_provenance":"INFERRED_single_image_size_depth_ambiguity","world_completeness":"partial",
                "control_level":obj.get("control_level","layout")},
            state={"matchable":True},confidence=obj["confidence"],provenance="INFERRED"),
            "Initialize uncertain coarse world from original image",[evidence_id])
    world.set_meta("camera",camera);world.set_meta("render_only",plan["render_only"])
    world.set_meta("relations",plan.get("relations",[]))
    for relation in plan.get("relations",[]):
        child=world.get_object(relation["subject"]);holder=world.get_object(relation["object"])
        x0,y0,x1,y1=child.attributes["initial_bbox_normalized"]
        depth=holder.position[1]
        child.position=[((x0+x1)*width/2-width/2)*depth/f,depth,
                        1.5-((y0+y1)*height/2-height/2)*depth/f]
        child.bbox_3d=[(x1-x0)*width*depth/f,child.bbox_3d[1],(y1-y0)*height*depth/f]
        child.attributes["geometry_provenance"]="INFERRED_shared_depth_with_holder_not_measured"
        child.attributes["holder_id"]=holder.object_id
        world.save(child,"Coarse held-object depth tied to visible holder; not precise articulated pose",
                   ["initial:"+child.object_id,"initial:"+holder.object_id])


def start_action(world,goal,plan):
    current=world.get_meta("active_action")
    if current and current["status"] not in {"COMPLETE","CANCELLED"}:
        raise ValueError("An unfinished action exists; resume or explicitly cancel it")
    if current:
        history=world.get_meta("action_history",[])
        history.append({"goal":current["goal"],"status":current["status"],
                        "accepted_segments":current["accepted_segments"],"attempts":current["attempts"]})
        world.set_meta("action_history",history[-20:])
    action={"goal":goal,"plan":plan,"phase_index":0,"status":"RUNNING","accepted_segments":0,
            "attempts":0,"history":[],"user_input_ready":False}
    world.set_meta("active_action",action)
    return action


def camera_look_at(camera,position,target):
    forward=np.asarray(target)-position;forward=forward/np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);norm=np.linalg.norm(right)
    if norm<1e-6:raise ValueError("Camera direction parallel to up")
    right/=norm;down=np.cross(forward,right)
    r=np.stack([right,down,forward]);t=np.eye(4);t[:3,:3]=r;t[:3,3]=-r@position
    return {**camera,"position":position.tolist(),"target":np.asarray(target).tolist(),"world_to_camera":t.tolist()}


def propose_timeline(world,phase,frames,fps=24):
    snapshot=world.snapshot();camera=snapshot["camera"]
    moves={m["object_id"]:np.array(m["delta_world"],float) for m in phase["moves"]}
    holders={r["subject"]:r["object"] for r in world.get_meta("relations",[]) if r["predicate"]=="held_by"}
    for child in holders:
        if child in moves:continue
        node=child;seen=set()
        while node in holders and node not in moves:
            if node in seen:raise ValueError("Cyclic held-object relation")
            seen.add(node);node=holders[node]
        moves[child]=moves.get(node,np.zeros(3)).copy()
    eye=np.array(camera["position"],float);target=np.array(camera["target"],float)
    rows=[]
    for i in range(frames):
        t=min(1,i/max(1,frames-1)/phase.get("motion_end_fraction",.65));a=t*t*(3-2*t)
        objects=deepcopy(snapshot["objects"])
        for obj in objects:
            obj["position"]=(np.array(obj["position"])+a*moves.get(obj["object_id"],np.zeros(3))).tolist()
            obj["state"]["control_enabled"]=obj["object_id"] in phase.get("controlled_object_ids",[o["object_id"] for o in objects])
        current_target=target
        if phase.get("camera_target_id"):
            desired=np.array(next(o["position"] for o in objects if o["object_id"]==phase["camera_target_id"]))
            current_target=(1-a)*target+a*desired
        current_camera=camera_look_at(camera,eye+a*np.array(phase["camera_delta"]),current_target)
        rows.append({"server_time_sec":world.get_meta("tick",0)+i/fps,"camera":current_camera,"objects":objects})
    return rows


def apply_assessment(world,decision,timeline,rollout_id):
    with world.transaction():
        return _apply_assessment(world,decision,timeline,rollout_id)


def _apply_assessment(world,decision,timeline,rollout_id):
    """Commit only completed intended transitions; observations remain in their store."""
    action=world.get_meta("active_action")
    if action is None or action["status"]!="RUNNING":raise ValueError("No running action")
    decision=deepcopy(decision)
    if decision.get("action") not in {"ACCEPT","REGENERATE"}:raise ValueError("Invalid assessment")
    if not isinstance(decision.get("confidence"),(int,float)) or not math.isfinite(decision["confidence"]) or not 0<=decision["confidence"]<=1:
        raise ValueError("Invalid decision confidence")
    for key in ("phase_complete","action_complete"):
        if type(decision.get(key)) is not bool:raise ValueError("Completion must be Boolean")
    if decision["action"]=="REGENERATE" and (decision["phase_complete"] or decision["action_complete"]):
        decision["phase_complete"]=False;decision["action_complete"]=False
    phase_index=action["phase_index"]
    if decision["action_complete"] and (not decision["phase_complete"] or phase_index!=len(action["plan"]["phases"])-1):
        decision["action_complete"]=False
    if decision["phase_complete"] and (not decision.get("completion_evidence") or decision.get("confidence",0)<.6):
        decision["phase_complete"]=False;decision["action_complete"]=False
    if phase_index==len(action["plan"]["phases"])-1 and not decision["action_complete"]:
        decision["phase_complete"]=False
    action["attempts"]+=1
    if decision["action"]=="ACCEPT":
        action["accepted_segments"]+=1
        # Partial visual progress does not prove the planned endpoint. Preserve world
        # geometry and retry/continue this phase instead of silently teleporting actors.
        if decision["phase_complete"]:
            for raw in timeline[-1]["objects"]:
                obj=WorldObject(**raw);obj.provenance="AGENT_CREATED"
                existing=world.get_object(obj.object_id)
                if existing is None:raise ValueError("Action cannot spawn generated-only objects")
                if np.allclose(existing.position,obj.position,rtol=0,atol=1e-9):
                    continue
                obj.state={**obj.state,"action_phase":phase_index,"transition":"intended_and_visually_assessed_not_measured"}
                world.save(obj,"Accepted intended action transition; not observation-derived geometry")
            world.set_meta("camera",timeline[-1]["camera"])
            world.set_meta("tick",timeline[-1]["server_time_sec"]+1/24)
            if phase_index<len(action["plan"]["phases"])-1:
                action["phase_index"]+=1
            elif decision["action_complete"]:
                action["status"]="COMPLETE";action["user_input_ready"]=True
    action["history"].append({"rollout_id":rollout_id,"phase_index":phase_index,"decision":decision})
    world.set_meta("active_action",action)
    return action
