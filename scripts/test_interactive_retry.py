import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_interactive_rollout import retry_available


class RetryTests(unittest.TestCase):
    def test_agent_rejection_and_unused_render_budget_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'result.json').write_text(json.dumps({'decision':'REGENERATE'}))
            call={'type':'tool_execution_start','toolName':'world','args':{'operation':'render'}}
            (root/'events.jsonl').write_text(json.dumps(call)+'\n')
            self.assertTrue(retry_available(root))
            (root/'events_retry.jsonl').write_text(json.dumps(call)+'\n')
            self.assertFalse(retry_available(root))

    def test_acceptance_does_not_reroll(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'result.json').write_text(json.dumps({'decision':'ACCEPT'}))
            self.assertFalse(retry_available(root))


if __name__=='__main__':unittest.main()
