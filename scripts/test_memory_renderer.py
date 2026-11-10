import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from renderer.comfy_h3 import ComfyH3Renderer
from renderer.memory_conditioning import image_bindings, memory_prompt


class MemoryRendererTests(unittest.TestCase):
    def test_binding_changes_but_first_frame_remains_distinct(self):
        first,history=object(),object()
        self.assertEqual(image_bindings(first),{'ref_image_1':first})
        self.assertEqual(image_bindings(first,history),{'ref_image_1':history,'ref_image_2':first})

    def test_prompt_numbering(self):
        text=memory_prompt('Orbit clockwise.',3,True)
        self.assertIn('<Picture 2> is the exact first RGB frame',text)
        self.assertIn('historical RGB on the left',text)
        self.assertIn('3.000-second',text)
        self.assertNotIn('<Picture 2>',memory_prompt('Orbit clockwise.',3,False))
        with self.assertRaises(ValueError):memory_prompt('Follow <Picture 1>',3,True)

    def test_real_command_contract_default_no_control(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def run(command,**kwargs):
                cases=json.loads(Path(command[command.index('--cases')+1]).read_text())
                self.assertEqual(cases[0]['controls'],[])
                self.assertIn('--lora',command)
                self.assertIn('--history-pair',command)
                self.assertIn('<Picture 2>',command[command.index('--prompt')+1])
                self.assertIn('--anchor-first',command)
                (root/'controlled.mp4').write_bytes(b'fake-for-command-test')
                return SimpleNamespace(returncode=0)
            with patch('renderer.comfy_h3.subprocess.run',side_effect=run):
                result=ComfyH3Renderer().render('Orbit clockwise.',None,root,first_frame='first.png',
                    reference_video='whitebox.mp4',history_pair='history.png',anchor_first=True)
            self.assertEqual(result,root/'controlled.mp4')
            self.assertEqual(json.loads((root/'generation_metadata.json').read_text())['control'],'disabled')

    def test_ambiguous_control_and_mixed_images_rejected(self):
        renderer=ComfyH3Renderer()
        with self.assertRaises(ValueError):renderer.render('x',None,'unused',control_video='depth.npz')
        with self.assertRaises(ValueError):renderer.render('x',None,'unused',history_pair='pair.png',appearance_images=['crop.png'])

if __name__=='__main__':unittest.main()
