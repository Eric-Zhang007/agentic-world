"""Agent edits a parameterized scene/action document, then sees its actual preview."""
import json
from pathlib import Path
from agent.action_planner import json_call


RESOURCE_PATH=Path(__file__).resolve().parents[1]/'configs/whitebox_resources.json'


def build(config,image,goal,feedback=None,previous=None,preview_images=()):
    return json_call(config,
        'You construct a persistent executable WHITEBOX world, not a video prompt. Return strict JSON. '
        'Use the provided parameterized resources. Model all interaction-relevant visible structures: '
        'ground, stairs, walls, important props, player, opponent, held weapon. Do not cap yourself at four objects. '
        'You may creatively fill unseen areas and add an offscreen NPC with a patrol, recording it as AGENT_CREATED. '
        'Visible geometry inferred from the image is INFERRED, not measured. Source may itself be a generated scene. '
        'World lore is creative and persistent. Preserve visible topology, relative scale and facing. '
        'Every actor needs a player/guard/patrol behavior; NPCs must execute behavior, never freeze because offscreen. '
        'Terrain: size [24,24], resolution 48, seed integer, amplitude 0.0..0.5, flatten_radius 8. '
        'Place active actors in flat central area at ground y=0. Box/stairs position is CENTER; '
        'place their bases at ground by y=size_y/2. Actors position is FOOT ROOT, size [width,height,depth]. '
        'Keep actor roots clear of static boxes, stair footprints and each other by body radii. '
        'Use <=1.1m body width, human heights around 1.8m and a plausible 2.0..2.8m staff. '
        'Staff requires holder and length, plus size [diameter,length,diameter]; its initial position may equal holder root. '
        'Coordinates are Godot Y-up. World +Z is not necessarily camera forward: compute image placement '
        'using camera position and target, and align all key static structures as well as actors. '
        'Choose facing and NPC guard/patrol policies consistent with the visible roles and user intention; '
        'creative offscreen additions are optional, never a requirement to add a particular species or NPC. '
        'Do not make a collider for tiny decoration. Gates are separate two pillars and lintel, not a solid impassable wall. '
        'Output {scene:{schema_version:1,seed:int,lore:string,terrain:{size:[w,d],resolution:int,seed:int,'
        'height_amplitude:float,flatten_radius:float},camera:{position:[x,y,z],target:[x,y,z]},'
        'entities:[{id:string,kind:actor|box|staff|stairs,category:string,position:[x,y,z],rotation_y:float,'
        'size:[x,y,z],color:[r,g,b],provenance:INFERRED|AGENT_CREATED,reason:string,description:string,'
        'behavior:optional object,holder:optional string,length:optional number,steps:optional int,'
        'appearance_bbox:optional [x0,y0,x1,y1]}]},'
        'action:{goal:string,actor_id:string,phases:[{type:move|sweep|recover|wait,duration_sec:number,'
        'criterion:string,delta:optional [x,0,z],start_angle_deg:optional number,end_angle_deg:optional number,'
        'camera_delta:optional [x,y,z]}]},assumptions:[string]}. '
        'Do not put optional irrelevant fields as null; omit them. Color in 0..1. '
        'Choose supported action phases appropriate to the actual user request. '
        'The sweep starts at 55deg default ready angle, ends -55deg, so no phase boundary teleport. '
        'The user should see the movement in the WHITEBOX preview. Preserve actor dimensions, identities and static scenery during actions. '
        'If previous proposal/feedback is supplied repair that proposal rather than invent an unrelated scene.',
        {'goal':goal,'resources':json.loads(RESOURCE_PATH.read_text()),'feedback':feedback,'previous':previous},
        [image]+list(preview_images))


def plan(config,scene,goal,feedback=None,previous=None,images=()):
    return json_call(config,
        'Plan an executable proxy-world action using existing persistent IDs. Return JSON action only. '
        'Do not create a new scene or move static geometry. NPCs already execute independent policies. '
        'Schema {goal:string,actor_id:string,phases:[{type:move|sweep|recover|wait|posture,duration_sec:number,'
        'criterion:string,delta:optional [x,0,z],start_angle_deg:optional number,end_angle_deg:optional number,'
        'camera_delta:optional [x,y,z],camera_follow_actor:optional existing actor ID,camera_distance_delta:optional metres 0..5,camera_orbit_yaw_deg:optional degrees -180..180,end_velocity:optional [vx,0,vz],target_posture:optional standing|crouched|low_guard}]}. All moves horizontal <=4m, durations .5..8s. '
        'camera_orbit_yaw_deg rotates the camera around its current target in the world Y-up plane, '
        'preserving radius and height throughout a smooth full-phase arc. It excludes camera_delta. '
        'Positive yaw uses the right-handed Y rotation; this is camera motion, never actor rotation. '
        'For an orbit/revisit request use an explicit substantial signed orbit angle, not a pan or dolly. '
        'camera_follow_actor translates camera position and target by that actor actual displacement, '
        'preserving initial relative framing throughout the phase. It excludes camera_delta. Use it when '
        'following is allowed and necessary for a full-body view; do not add it to fixed-camera requests. '
        'camera_distance_delta smoothly increases radial distance over the entire phase without changing '
        'the initial camera or target. Alone it follows the initial view direction; combined with orbit '
        'it follows the rotated current radial direction, so radius and height may increase. '
        'Combine with follow to add framing clearance. '
        'For continuous motion specify end_velocity in world m/s for EVERY move. This uses a cubic root path '
        'starting with the actor motion.velocity and ending with end_velocity, with no automatic settling. '
        'A missing initial motion means a legacy stationary boundary. Nonzero velocity cannot jump to wait; '
        'use a move ending at zero velocity to stop. Respect the exact available generation window and '
        'requested intermediate moving boundary. The end of a video is not the end of a user action. '
        'The posture phase changes coarse support state without root translation. low_guard uses feet and both '
        'hands on the ground and cannot hold a staff. Rise before locomotion from crouched or low_guard; '
        'crouched gait is not implemented. Existing posture '
        'weights persist during wait and into the next segment; do not reset them implicitly. '
        'Do not invent constraints on exact grip count, hand position, or joint angles when the user only asks '
        'to retain a weapon. One-handed and two-handed grips preserve ownership; secondary acting is free. '
        'Sweeps have 45..200deg extent, angles in [-150,150], start at current weapon_angle (default55). '
        'Keep contact separate from damage. Repair failed action based on feedback without weakening the goal.',
        {'scene':scene,'goal':goal,'resources':json.loads(RESOURCE_PATH.read_text()),'feedback':feedback,'previous':previous},images)


def review(config,scene,action,report,images,source_image=None,continuation=False):
    return json_call(config,
        'You are reviewing actual rendered WHITEBOX previews, not H3. Return strict JSON '
        '{decision:ACCEPT|REGENERATE,evidence:[string],issues:[string],suggested_changes:[string]}. '
        'Accept only if the preview clearly demonstrates the requested coarse action, actors stay grounded, '
        'weapon remains held, NPCs have plausible independent motion and static scene persists. '
        'Geometry is intentionally simple: do not reject missing texture, fur, face, fingers or exact costume. '
        'Use the original as visual layout evidence, but allow disclosed creative offscreen additions. '
        'Review terrain, barriers, relative spatial arrangement, body/weapon motion and history consistency. '
        'Numerical checks alone do not establish visual correctness; visual acceptance cannot override numerical violations. '
        'Do not infer damage from proxy_contact. If uncertain say so; do not fabricate hidden events. '
        'Images after optional source are actual multi-view and chronological preview frames. '
        'The last three frames per patrol NPC are separate close-up chronological views proving offscreen behavior. '
        'On initial construction compare source actor left/right ordering; do not approve a mirrored reconstruction. '
        'On continuation use the committed scene and prior action history instead; earlier accepted changes need not be undone. '
        'Ground contact is checked from feet and terrain, not shadow separation alone; directional shadows can be offset.',
        {'scene':scene,'action':action,'numerical_report':report,'source_first':bool(source_image),'continuation':continuation,
         'review_image_order':[str(p) for p in images]},
        ([source_image] if source_image else [])+list(images))
