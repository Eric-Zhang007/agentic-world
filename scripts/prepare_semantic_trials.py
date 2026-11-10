"""Place declared interaction fixtures into the recorded GTA street branch."""
import json
from copy import deepcopy
from pathlib import Path
from whitebox.motion import prepare_scene,validate_geometry
from scripts.test_action_semantics import obstacle


def main():
    root=Path('outputs/semantic_actions').resolve();root.mkdir(parents=True,exist_ok=False)
    seed=json.loads(Path('outputs/gta_asset_alignment/surface_motion_scene.json').read_text())
    p=seed['entities'][0]['position'];x,y,z=p
    for name in ('crouch','jump'):
        target=root/name;target.mkdir()
        scene=deepcopy(seed)
        scene['entities'][0]['rotation_y']=1.5707963267948966
        scene['camera']={'position':[x+3,y+2.8,z+5],'target':[x+1,y+.8,z],'fov':50}
        scene['entities'].append(obstacle(name+'_obstacle',
            [x+.9,y+1.55,z] if name=='crouch' else [x+1,y+.2,z],
            [.6,.5,2] if name=='crouch' else [.3,.4,2]))
        scene=prepare_scene(scene)
        if validate_geometry(scene):raise ValueError(validate_geometry(scene))
        (target/'scene.json').write_text(json.dumps(scene,indent=2))
    prompt=f'''Work in /home/star/agentic_world. This is a whitebox action planning test on recorded GTA street geometry. Do not call H3, render, any gateway directly, or alter implementation/assets/scenes. Use the registered world tool. Read configs/whitebox_resources.json and harness/pi/world_instructions.md.
There are two independent scenes under {root}: crouch/scene.json and jump/scene.json. The extra beam/barrier is intentionally AGENT_CREATED, not claimed to be present in the GTA recording.
For each scene initialize a NEW world.sqlite in that case directory, query status, author action.json and call plan to a new plan directory. Revise rejected plans using the error feedback. Keep the existing scenery and camera. The user goals are:
1. Crouch case: duck under the overhead beam, move 1.8 metres in world +X, then stand fully beyond it. Do not walk around it or move the beam.
2. Jump case: jump over the low barrier, land 2 metres in world +X, and continue walking another metre without stopping at landing. End at rest. Do not delete or walk around the barrier.
Each whole plan must fit 3 seconds / 73 video frames at 24fps. Choose phase durations and velocities yourself within executable constraints. No joint coordinates. Use prospective plan only; do not commit or claim video acceptance. Once both plans pass, write a brief result.json containing actual plan paths, rejected attempts if any, and what has/has not been verified. Stop after planning; an external test will compile and visually inspect the whitebox. Do not create an independent model-calling script.'''
    (root/'agent_prompt.txt').write_text(prompt,encoding='utf-8')
    print(root)


if __name__=='__main__':main()
