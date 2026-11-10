"""Protect expensive ablations from silently changing unrelated render inputs."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_reference_ablation import main


class ReferenceAblationTests(unittest.TestCase):
    def test_strength_only_override_preserves_exact_prompt_seed_and_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            command=['python','infer.py','--prompt','old','--output','old/render',
                     '--strength','1.0','--seed','102','--steps','40','--control-video','depth.npz',
                     '--reference-image','boundary.png','--cache-threshold','0.0']
            record=root/'record.json';record.write_text(json.dumps({'command':command}))
            prompt=root/'prompt.txt';prompt.write_text('Exact memory text\nSecond line.',encoding='utf-8')
            args=argparse.Namespace(record=record,prompt_file=prompt,appearance_indices=[],
                                    output=root/'out',gpu=1,strength=.5)
            with patch('scripts.run_reference_ablation.subprocess.run') as run:
                run.return_value.returncode=0
                with self.assertRaises(SystemExit) as end:main(args)
            self.assertEqual(end.exception.code,0)
            actual=run.call_args.args[0]
            expected=list(command)
            expected[expected.index('--output')+1]=str((root/'out/render').resolve())
            expected[expected.index('--prompt')+1]=prompt.read_text()
            expected[expected.index('--strength')+1]='0.5'
            self.assertEqual(actual,expected)
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'1')
            metadata=json.loads((root/'out/ablation.json').read_text())
            self.assertEqual(metadata['control_strength_override'],{'before':1.,'after':.5})
            self.assertFalse(metadata['world_commit'])

    def test_invalid_strength_does_not_launch_renderer(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            record=root/'record.json';record.write_text(json.dumps({'command':[
                'python','infer.py','--prompt','old','--output','old','--strength','1']}))
            prompt=root/'prompt.txt';prompt.write_text('test')
            for value in [-1,float('nan'),float('inf')]:
                args=argparse.Namespace(record=record,prompt_file=prompt,appearance_indices=[],
                                       output=root/'out',gpu=0,strength=value)
                with patch('scripts.run_reference_ablation.subprocess.run') as run:
                    with self.assertRaises(ValueError):main(args)
                    run.assert_not_called()
                self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
