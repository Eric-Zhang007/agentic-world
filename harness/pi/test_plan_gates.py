import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from world_tools import dispatch,check_plan,save
from whitebox.motion import simulate
from scripts.test_whitebox_world import fixture


class PlanGateTests(unittest.TestCase):
    def plan(self,path):
        scene=fixture()
        action={'actor_id':'player','phases':[{'type':'wait','duration_sec':3,'criterion':'Observe'}]}
        timeline,_,_=simulate(scene,action)
        for name,data in [('scene',scene),('action',action),('timeline',timeline),('status',{'world_start':0})]:
            save(path/(name+'.json'),data)
        return scene,timeline

    def test_compile_rejects_edited_timeline_before_rendering(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);_,timeline=self.plan(p)
            timeline['frames'][0]['entities'][0]['position'][0]+=2
            save(p/'timeline.json',timeline)
            with patch('world_tools.subprocess.run') as renderer:
                with self.assertRaisesRegex(ValueError,'PLAN_CHANGED'):
                    dispatch('compile',{'plan':str(p),'output':str(p/'compiled')})
                renderer.assert_not_called()
            self.assertFalse((p/'compiled').exists())

    def test_render_requires_compiler_output(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'COMPILE_REQUIRED'):
                dispatch('render',{'request':{'reference_video':str(Path(folder)/'whitebox.mp4'),
                    'first_frame':'frame.png','controls':[]}})

    def test_render_rejects_changed_scene_even_with_valid_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);scene,timeline=self.plan(p)
            video=p/'whitebox.mp4'
            save(p/'compiled_plan.json',{'plan':str(p),'scene':scene,'timeline':timeline,'reference_video':str(video)})
            scene['camera']['fov']=45
            save(p/'scene.json',scene)
            import json
            action=json.loads((p/'action.json').read_text())
            changed,_,_=simulate(scene,action)
            save(p/'timeline.json',changed)
            with self.assertRaisesRegex(ValueError,'RECOMPILE_REQUIRED'):
                dispatch('render',{'request':{'reference_video':str(video),'first_frame':'frame.png','controls':[]}})

    def test_unchanged_plan_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);scene,timeline=self.plan(p)
            self.assertEqual(check_plan(p),(scene,timeline))

    def test_saved_success_cannot_override_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);scene,_=self.plan(p)
            next(e for e in scene['entities'] if e['id']=='wall')['position']=[0,1,2]
            action={'actor_id':'player','phases':[{'type':'move','duration_sec':3,
                'delta':[0,0,4],'criterion':'Walk through wall'}]}
            timeline,_,_=simulate(scene,action)
            save(p/'scene.json',scene);save(p/'action.json',action);save(p/'timeline.json',timeline)
            save(p/'diagnostics.json',{'valid':True})
            with self.assertRaisesRegex(ValueError,'PLAN_REJECTED'):check_plan(p)


if __name__=='__main__':unittest.main()
