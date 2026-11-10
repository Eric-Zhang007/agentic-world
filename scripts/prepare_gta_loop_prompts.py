"""Freeze action-only trials before looking at any generated continuation."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gta_pi_loop_20261102'
REMOTE='/home/star/agentic_world/outputs/gta_pi_loop_20261102'
ACTIONS={
    'rural':[
        'Walk a couple of steps toward the other person and stop with comfortable personal space. The other person continues a slow natural walk.',
        'Move to the clear side of the other person without touching them; let the camera gently orbit about eight degrees.',
        'Stop near the road edge and wait while the other person continues naturally. Keep the storefront, palm trees and roadside poles unchanged.'
    ],
    'urban':[
        'Walk a couple of natural steps forward along the brick road and stop, retaining the other pedestrian and street landmarks.',
        'Move to the right side of the clear road to leave room for the other pedestrian. Let the camera gently orbit about ten degrees while keeping both people in view.',
        'Walk back toward your original starting area near the lamppost and stop. Preserve the street layout and the identity of both people.'
    ],
    'harbor':[
        'Walk a couple of steps toward the parked off-road vehicle and stop before touching it. The other pedestrian continues walking naturally.',
        'Walk around the near left corner of the parked vehicle, staying clear of it and the other pedestrian. Keep the camera following gently.',
        'Stop alongside the parked vehicle. Let the camera make a small orbit to show the player, vehicle and nearby pedestrian together.'
    ]}

def main():
    for case,actions in ACTIONS.items():
        folder=REMOTE+'/'+case
        prompt=f'''Continue this same Pi session from its initialized world at {folder}/world.sqlite.
This is an autonomous rollout trial. The only new user intentions are these,
in order: {json.dumps(actions)}
Do not modify source code, change system constraints or call another model.
Do not read dataset ground truth. Use existing world tools and render skill.
No human feedback will be supplied. You must inspect actual video images yourself.
Maximum FOUR H3 generations TOTAL in this trial; maximum TWO attempts for a
given unaccepted segment. Aim for three accepted 3-second segments. If a goal is
partially complete, continue its remaining action before moving to the next user
intention; never label it complete to advance the list. Stop and report a failure
when available state representation cannot describe a consequential change.

For every segment: status -> write action -> plan -> compile -> inspect proxy
opening/end -> write prompt -> render -> inspect_video -> read RGB frames ->
write review -> commit -> status. Use new directories for every attempt.
Record segment, user goal index, action completion, retries and reasons in
trial_summary.json as you go. Do not commit if you have not looked at the images.
Use world compile with width=1024 height=576. Read its proxy/camera.png and last
frame image. The plan is NOT video evidence. Keep first_frame=input.png for the
first rollout, then the last actual RGB frame returned by accepted evidence.
Use renderer options gpu={dict(urban='0',harbor='1',rural='2')[case]} (string), steps=40,
frames=73, width=1024, height=576, timeout=2400, kernel_backend=default.
Each action should sum to 3 seconds (72 ticks) so plan returns 72 frames and
compiled video has 73. If plan alignment returns another count use its count
for renderer.frames; never stretch the motion or silently truncate frames.
world render request: prompt authored by you, source_video=null,
output_dir=NEW segment/render directory, first_frame=current actual RGB,
reference_video=compiled whitebox.mp4, anchor_first=true, seed=43+attempt,
controls=[{{path:compiled depth.npz,strength:0.5,end:1.0}}],
appearance_images=[], cache_threshold=0.0. Do not feed whitebox as RGB appearance.
No extra memory images; identity comes from current RGB and concise text.
The whitebox video controls trajectory, layout and camera; its primitive geometry,
colors and mannequin appearance must NOT be reproduced. Describe visible people,
clothes and landmarks succinctly, not internal object IDs or coordinate vectors.
Prompt must include: 起始和结束状态应与给定条件一致，中间按动作描述自然丰富；收势是在目标终态稳定下来，不意味着回到起始位置。
Read skills/render-rollout/SKILL.md when composing prompts.

Inspect evidence every 0.5 seconds, explicitly opening first, middle, last and
any additional needed images. Judge identity, background continuity, intended
movement, camera, contact/obstacle clearance and attained endpoint separately.
Unknown metric depth is uncertain, not automatic failure or an invented precise
measurement. Correct root/posture/camera from actual evidence when supported;
use UNOBSERVED_SIMULATED for unseen NPCs. If the video loses identities, turns
people into whitebox figures, or changes key topology, reject rather than accept.
If motion is partial but consistent, accept the ACTUAL attained state and continue.
Stop after three accepted segments or generation budget. Save a final summary
including which capabilities failed, elapsed time if available, and artifact paths.
'''
        (OUT/case/'loop_prompt.txt').write_text(prompt,encoding='utf-8')
        if case=='rural':
            debug=('This is a separate LOW-RESOLUTION orchestration trial: 512x288, 20 steps. '
                   'Do not use it to claim high-resolution visual quality. All other acceptance and history rules still apply.\n'+prompt)
            debug=debug.replace('width=1024 height=576','width=512 height=288').replace('steps=40','steps=20').replace('width=1024, height=576','width=512, height=288')
            (OUT/case/'loop_debug_prompt.txt').write_text(debug,encoding='utf-8')
        if case=='harbor':
            resumed='''Framework update: prospective NPC policies are now supported in action JSON as
npc_behaviors:[{id,behavior,reason}]. Read harness/pi/world_instructions.md for
the generic interface. You choose all behavior parameters. No world state or
previous plan has been edited. Preserve the previous trial_summary.json as
baseline_representation_failure.json before retrying. Keep this as a separately
reported post-fix trial, not a successful original baseline.\n'''+prompt
            (OUT/case/'loop_npc_prompt.txt').write_text(resumed,encoding='utf-8')
        (OUT/case/'user_actions.json').write_text(json.dumps(actions,indent=2),encoding='utf-8')
    print('Frozen urban and harbor trial instructions')

if __name__=='__main__':main()
