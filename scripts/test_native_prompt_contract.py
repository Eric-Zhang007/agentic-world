import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent.render_contract import compile_prompt


class NativePromptContractTests(unittest.TestCase):
    def test_native_video_and_empty_memory_are_explicit(self):
        entity={'id':'actor','category':'person','position':[0,0,0]}
        scene={'entities':[entity],'camera':{'position':[0,1,4],'target':[0,1,0]}}
        entity['kind']='actor'
        action={'actor_id':'actor'}
        timeline={'fps':24,'frames':[scene]}
        def reply(config,system,contract,images):
            self.assertEqual(contract['available_appearance_object_ids'],[])
            self.assertEqual(contract['native_video_reference']['tag'],'<Video 1>')
            self.assertEqual(contract['native_video_reference']['path'],'proxy.mp4')
            return {'prompt':'Use <Video 1> for the camera and <Picture 1> for appearance.',
                    'appearance_object_ids':[]}
        with tempfile.TemporaryDirectory() as folder,patch('agent.render_contract.json_call',side_effect=reply):
            prompt=compile_prompt({},scene,action,timeline,'first.png',Path(folder),reference_video='proxy.mp4')
            self.assertIn('<Video 1>',prompt)


if __name__=='__main__':unittest.main()
