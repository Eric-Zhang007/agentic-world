import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from scripts.freeze_prompt_baseline import main as freeze
from scripts.prepare_prompt_regression import HEADER, cases, command_for
from scripts.run_prompt_regression import validated_cases


class PromptRegressionTests(unittest.TestCase):
    def setUp(self):
        self.lines = ['Armored monkey first', 'Long rigid dark staff first',
                      'Stocky orange tiger first', 'Monkey-like warrior detail',
                      'Long straight staff detail', 'Broad, stocky tiger detail']
        self.base = 'action\nboundary\n' + HEADER + ' preserve\n' + '\n'.join(self.lines)
        self.changed = 'different action\n' + HEADER + '\n' + '\n'.join([
            'Carved stone monument detail', 'Weathered stone retaining detail',
            'Weathered temple terrace detail'])

    def test_only_requested_detail_is_removed(self):
        variants = cases(self.base, self.changed)
        self.assertEqual(variants['baseline'], self.base)
        for name, index in [('remove_player_detail', 3), ('remove_staff_detail', 4),
                            ('remove_tiger_detail', 5)]:
            removed = self.lines[index] + ('\n' if index < 5 else '')
            self.assertEqual(variants[name], self.base.replace(removed, '', 1))

    def test_architecture_is_appended_without_rewriting_baseline(self):
        value = cases(self.base, self.changed)['append_architecture']
        self.assertTrue(value.startswith(self.base + '\n'))
        self.assertNotIn('different action', value)

    def test_unrecognized_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            cases(self.base.replace('Long straight staff', 'other staff'), self.changed)

    def test_incomplete_architecture_is_rejected(self):
        for suffix in ['', '\nCarved stone monument detail']:
            with self.assertRaises(ValueError):
                cases(self.base, HEADER + suffix)

    def test_runner_rejects_coordinated_prompt_and_command_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = ['python', 'render.py', '--prompt', self.base, '--output', 'old', '--seed', '102']
            (root/'source_changed.json').write_text(json.dumps({'command': command_for(command, self.changed, 'x')}))
            for name, prompt in cases(self.base, self.changed).items():
                directory = root/name
                directory.mkdir()
                (directory/'prompt.txt').write_text(prompt)
                (directory/'prepared.json').write_text(json.dumps({'command': command_for(command, prompt, 'x')}))
            baseline = {'command': command, 'prompt': self.base}
            self.assertEqual(len(validated_cases(baseline, root, root/'out')), 5)
            directory = root/'remove_player_detail'
            wrong = 'unrelated rewrite'
            (directory/'prompt.txt').write_text(wrong)
            (directory/'prepared.json').write_text(json.dumps({'command': command_for(command, wrong, 'x')}))
            with self.assertRaisesRegex(ValueError, 'named ablation'):
                validated_cases(baseline, root, root/'out')

    def test_command_preserves_references_and_all_unknown_flags(self):
        command = ['python', 'render.py', '--prompt', 'old', '--output', 'old-dir',
                   '--appearance-image', 'a.png', '--seed', '102', '--other-flag']
        value = command_for(command, 'new', 'new-dir')
        self.assertEqual(value[:2] + value[6:], command[:2] + command[6:])
        self.assertEqual(command[3], 'old')
        self.assertEqual(value[3], 'new')

    def test_freeze_requires_completion_and_identity_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(metadata=root/'executed.json', prompt=root/'prompt.txt',
                                   completion=root/'exit.json', review=root/'review.json',
                                   name='test', output=root/'frozen.json')
            args.metadata.write_text(json.dumps({'command': ['python', '--prompt', self.base]}))
            args.prompt.write_text(self.base)
            args.review.write_text(json.dumps({'identity_consistency': {'status': 'SUPPORTED'}, 'overall': 'FAIL'}))
            args.completion.write_text(json.dumps({'returncode': 1}))
            with self.assertRaisesRegex(ValueError, 'did not complete'):
                freeze(args)
            args.completion.write_text(json.dumps({'returncode': 0}))
            args.review.write_text(json.dumps({'identity_consistency': {'status': 'CONTRADICTED'}}))
            with self.assertRaisesRegex(ValueError, 'not supported'):
                freeze(args)
            args.review.write_text(json.dumps({'identity_consistency': {'status': 'SUPPORTED'}, 'overall': 'FAIL'}))
            freeze(args)
            self.assertEqual(json.loads(args.output.read_text())['overall_assessment'], 'FAIL')
            with self.assertRaises(FileExistsError):
                freeze(args)


if __name__ == '__main__':
    unittest.main()
