"""Bounded image-to-coarse-world planning and multi-rollout action assessment."""
import base64
import io
import json
from pathlib import Path

from PIL import Image
from agent.client import AgentClient


def image_content(path):
    image=Image.open(path).convert("RGB");image.thumbnail((960,640))
    out=io.BytesIO();image.save(out,format="JPEG")
    return {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+
        base64.b64encode(out.getvalue()).decode()}}


def json_call(config, system, payload, images=()):
    content=[{"type":"text","text":json.dumps(payload,ensure_ascii=False)}]
    content.extend(image_content(path) for path in images)
    message=AgentClient(config).complete([{"role":"system","content":system},
                                         {"role":"user","content":content}])
    text=(message.get("content") or "").strip()
    if text.startswith("```"):
        text="\n".join(text.splitlines()[1:-1])
    result=json.loads(text)
    if not isinstance(result,dict):raise ValueError("Agent must return a JSON object")
    return result


def construct_and_plan(config,image,goal,segment_seconds,max_segments=3,existing_world=None):
    return json_call(config,
        "You are the world director of a persistent coarse 3D simulation, not a mesh artist. "
        "Return strict JSON. Separate visible evidence, uncertain geometry, intended actions, and renderer freedom. "
        "Only enumerate 1-4 important visible actors/objects. Never invent a visible actor just because the user mentions it. "
        "Do not infer exact physics, metric depth, enemy death, or health from one image. "
        "Use English render prompts. Respect the visible weapon: a staff user should not inexplicably drop their staff. "
        "Break a complex action into physically plausible chronological phases; never compress a long combination into one short clip. "
        "The user's action can remain RUNNING across clips. Each phase needs a visually checkable completion criterion. "
        "Fine poses, fur, cloth, particles and background details are RENDER_ONLY. "
        "CRITICAL granularity: completion criteria must reflect USER-requested outcomes, not extra joint-level choreography. "
        "Do not require frozen shoulders during a head turn, an exact grip/weapon angle during a sidestep, or invented millisecond deadlines. "
        "Natural secondary motion is allowed. A sidestep phase needs visible displacement, balance and weapon retention, not a specific guard pose unless the user required it there. "
        "Coordinate convention: X right, Y initially forward, Z up, approximate meters inferred from subject size. "
        "Camera may move, orbit, approach, or change view; prefer a continuous modest move for first tests because unseen geometry is uncertain. "
        "JSON schema: {scene_description:string, assumptions:[string], render_only:[string], "
        "objects:[{object_id:string,category:string,description:string,bbox_normalized:[x0,y0,x1,y1], "
        "approx_height:number,confidence:number,control_level:layout|identity_only}], "
        "relations:[{subject:string,predicate:held_by,object:string}], phases:[{name:string,render_prompt:string, "
        "duration_sec:number,motion_end_fraction:number,moves:[{object_id:string,delta_world:[dx,dy,dz]}], "
        "camera_delta:[dx,dy,dz],camera_target_id:string|null,completion_criterion:string, "
        "required_outcomes:[string],render_only_preferences:[string]}], "
        "action_completion_criterion:string}. Boxes must refer to the input image, coordinates in [0,1]. "
        "Category should be a simple detectable label such as person, tiger, vehicle. Each phase is one full segment; "
        "choose duration_sec from allowed_segment_seconds: short for a simple beat, longer for a dodge/contact/recovery that needs time. "
        "Use identity_only for articulated held props such as a staff; do not force the weapon into a moving cuboid. "
        "Record clearly visible held_by relations. Actor bodies may use layout. "
        "motion_end_fraction is between .3 and .9: finish displacement then hold so settling can be seen. "
        "do not plan more phases than max_segments. If the request cannot finish within this budget say so in assumptions; don't claim a win. "
        "If existing_world is supplied, retain exactly its object IDs and categories; plan moves relative to its current positions. "
        "Its existing geometry will be retained, not reinitialized from your image proposals.",
        {"goal":goal,"segment_seconds":segment_seconds,"max_segments":max_segments,
         "allowed_segment_seconds":sorted(set([segment_seconds,73/24,124/24])),
         "existing_world":existing_world,
         "mode":"single_image_inferred_world; metric geometry is not measured"},[image])


def assess_phase(config,goal,phase,report,frames,original_image,final_phase,completed_phases=None,committed_reference=None):
    return json_call(config,
        "Assess the rendered action against its intention and the original source image. Return strict JSON. "
        "You see the original source, then the last committed continuity reference, then ordered rollout frames. "
        "Judge NEW movement and continuity against the committed reference, not the episode's initial pose. "
        "The initial source provides identity/provenance context; earlier accepted moves need not be repeated or undone. "
        "Deterministic matching handles identity; discuss uncertainty rather than inventing matches. "
        "The scene is a partial, inferred coarse world: background extras are not automatically hallucinations. "
        "Decide ACCEPT or REGENERATE. ACCEPT means this clip can advance continuity, not that every requested action completed. "
        "Return phase_complete separately, true only if its completion criterion is visibly met. "
        "If motion makes progress but is unfinished you may ACCEPT with phase_complete=false and give a continuation prompt. "
        "Never mark an unexecuted future phase complete. The continuation prompt must contain ONLY the current phase, never a later attack or recovery phase. "
        "Do not declare combat victory just from a prompt or elapsed duration. "
        "If evidence is ambiguous say so. Do not add generated entities to world facts. "
        "Judge required_outcomes at coarse action level. render_only_preferences are not mandatory gates. "
        "Natural shoulder/arm follow-through is not failure of a head turn. Do not invent precise pose requirements. "
        "Do not use elapsed time alone as evidence, but do not demand an exact hidden foot contact when the visible requested outcome is clear. "
        "completed_phases lists earlier accepted, visually assessed milestones. Do not require those earlier actions to repeat in the current clip. "
        "For final action completion combine that ordered ledger with the current phase evidence; do not invent missing prior milestones. "
        "JSON: {action:ACCEPT|REGENERATE,phase_complete:boolean,action_complete:boolean, "
        "confidence:number,reason:string,completion_evidence:[string],continuation_prompt:string, "
        "unexpected_details:[string]}. action_complete requires final_phase and phase_complete and observed goal completion. "
        "No proposed world edits: accepted intended transitions are applied separately with AGENT_CREATED provenance.",
        {"goal":goal,"phase":phase,"discrepancy":report,"final_phase":final_phase,
         "completed_phases":completed_phases or [],
         "frame_timestamps":[f["timestamp"] for f in frames]},
        [original_image,committed_reference or original_image]+[f["path"] for f in frames])


def refine_phase(config,phase,decision,world,image):
    summary=world.snapshot()
    for obj in summary["objects"]:obj.pop("appearance_embedding",None)
    return json_call(config,
        "Revise ONLY the current action phase after visual feedback. Return a JSON phase with the same "
        "name, duration_sec and completion_criterion, plus render_prompt, motion_end_fraction, moves, controlled_object_ids, required_outcomes, render_only_preferences, "
        "camera_delta, camera_target_id. The world and image are the last COMMITTED state; the failed/partial "
        "video endpoint has NOT been committed. Restart from this state, not an imagined advanced endpoint. "
        "Never include actions belonging to later phases, even if the review suggestion mentions them. "
        "controlled_object_ids selects only coarse bodies needing layout; omit fine articulated held props such as a staff. "
        "Use a modest camera move or lock it if camera motion confounded assessment. Do not change IDs. "
        "Preserve required_outcomes; do not promote render_only_preferences into mandatory conditions. "
        "Use motion_end_fraction .3-.9 to leave a visible settling interval. Do not weaken the completion criterion.",
        {"phase":phase,"feedback":decision,"world":summary},[image])
