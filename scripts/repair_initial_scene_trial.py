"""Ask the existing whitebox director to repair an unaccepted initial scene."""
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from whitebox import director


def main():
    root=ROOT/'outputs/aligned_world_20261014'
    scene=json.loads((root/'camera_fit/scene.json').read_text())
    action=json.loads((root/'attempt_001/action.json').read_text())
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())['agent']
    result=director.build(cfg,ROOT/'outputs/aligned_seed_20261014_b/frame_011.png',
        'Repair this NEW UNACCEPTED initialization. No old world/history is being changed. '
        'Keep the fitted camera and actor roots/heights which align image head/feet markers. '
        'Align the initial staff visible direction, placement and grip with the image; do not simply '
        'keep a default procedural grip. Describe any representation limit rather than pretending it is fixed. '
        'Match major stairs, plinths and statue masses in projection. Only a .5 second wait, fixed camera. '
        'Tiger guard should stay at its initial satisfied distance; actual schema is target (not target_id), '
        'distance (not maintain_distance), strafe numeric speed. Do not make tiger advance during hold. '
        'Actor weapon_elevation_deg may vary -90..90 and grip is left/right/both. '
        'No new characters and no fine ornament reconstruction.',
        feedback={'issues':['Prior camera too wide: corrected camera supplied now.',
            'Staff upper tip outside frame and lower end too high; source upper tip near [36,150], lower tip cropped at bottom.',
            'Source holding hand near [220,430], coarse proxy hand too far right.',
            'Static terrace colliders cover the central stair approach. Restore visible open staircase.'],
            'landmarks':json.loads((root/'landmarks.json').read_text())},
        previous={'scene':scene,'action':action},
        preview_images=[root/'camera_fit/preview/camera.png'])
    (root/'repair_proposal.json').write_text(json.dumps(result,indent=2))
    print('Saved repair_proposal.json',flush=True)


if __name__=='__main__':main()
