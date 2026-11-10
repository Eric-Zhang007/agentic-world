import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness.pi.world_tools import dispatch,save,ROOT
from scripts.test_whitebox_world import fixture


class ExtensionTests(unittest.TestCase):
    def test_addition_preserves_existing_state_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=fixture();save(root/'scene.json',scene);db=str(root/'world.sqlite')
            dispatch('initialize',{'scene':str(root/'scene.json'),'db':db})
            entity=deepcopy(scene['entities'][-1]);entity.update(id='new_crate',position=[8,1,-10],size=[1,2,1])
            save(root/'edits.json',[{'op':'spawn','entity':entity}])
            result=dispatch('extend',{'db':db,'revision':0,'edits':str(root/'edits.json'),
                'reason':'Extend unseen side of street','output':str(root/'extension')})
            state=dispatch('status',{'db':db})
            self.assertEqual(result['revision_after'],1);self.assertEqual(state['tick'],0.)
            self.assertEqual(state['history_length'],0)
            self.assertEqual(state['scene']['entities'][:-1],scene['entities'])
            with self.assertRaisesRegex(ValueError,'changed'):
                dispatch('extend',{'db':db,'revision':0,'edits':str(root/'edits.json'),
                    'reason':'Stale','output':str(root/'stale')})

    def test_cannot_spawn_into_initial_rgb_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=fixture();save(root/'scene.json',scene);db=str(root/'world.sqlite')
            dispatch('initialize',{'scene':str(root/'scene.json'),'db':db})
            before=dispatch('status',{'db':db})
            entity=deepcopy(scene['entities'][-1]);entity.update(id='new_crate',position=[4,1,2],size=[1,2,1])
            save(root/'edits.json',[{'op':'spawn','entity':entity}])
            with self.assertRaisesRegex(ValueError,'starting view'):
                dispatch('extend',{'db':db,'revision':0,'edits':str(root/'edits.json'),
                    'reason':'Wrong insertion','output':str(root/'extension')})
            self.assertEqual(before,dispatch('status',{'db':db}))


if __name__=='__main__':unittest.main()
