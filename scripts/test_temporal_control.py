import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from renderer.temporal_control import frame_spans, temporal_schedule, gate_tensors


class TemporalControlTest(unittest.TestCase):
    def test_exact_native_frame_partition(self):
        for count in (5,22,73,124):
            spans = frame_spans(count)
            self.assertEqual([i for a,b in spans for i in range(a,b)], list(range(count)))
            self.assertEqual(len(spans), (count-5)//17*5+2)

    def test_endpoint_only(self):
        schedule = temporal_schedule(73,[0,72])
        self.assertEqual(schedule["active_frame_indices"], [0,69,70,71,72])
        self.assertEqual(sum(schedule["latent_weights"]), 2)

    def test_dense(self):
        self.assertEqual(temporal_schedule(73,None)["active_fraction"],1)

    def test_invalid(self):
        for value in ([],[-1],[73],[1.5],[True]):
            with self.assertRaises(ValueError): temporal_schedule(73,value)
        with self.assertRaises(ValueError): frame_spans(75)

    def test_no_residual_to_reference_audio_text_or_middle(self):
        import torch
        latent = torch.ones(1,24,22,2,2)
        layout = SimpleNamespace(seq_len=54,segments=[(0,4,"text"),(4,6,"ref_img"),
                                                     (6,10,"audio"),(10,54,"video")])
        masked, rows = gate_tensors(temporal_schedule(73,[0,72]),latent,layout)
        self.assertEqual(rows[:10].sum().item(),0)
        self.assertEqual(rows.nonzero().flatten().tolist(),[10,11,52,53])
        self.assertEqual(masked[:,:,1:-1].count_nonzero().item(),0)


if __name__ == "__main__": unittest.main()
