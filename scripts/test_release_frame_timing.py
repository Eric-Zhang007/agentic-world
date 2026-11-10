import tempfile
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import MagicMock,patch
from types import SimpleNamespace
import av
import numpy as np
from scripts.finalize_gta_release import video_shape


def write_clip(path,pts,bframes=0):
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=24)
        stream.width=stream.height=32
        stream.pix_fmt='yuv420p'
        stream.options={'bf':str(bframes),'crf':'18'}
        for i,t in enumerate(pts):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),i*20,dtype=np.uint8),format='rgb24')
            frame.pts=t
            frame.time_base=Fraction(1,24)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


class ReleaseTimingTests(unittest.TestCase):
    def test_cfr_and_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'normal.mp4'
            write_clip(path,range(6))
            endpoints=[]
            self.assertEqual(video_shape(path,endpoints,fps=24),(6,32,32,3))
            self.assertEqual(len(endpoints),2)
            self.assertLess(endpoints[0].mean(),endpoints[1].mean())

    def test_duplicate_pts_not_hidden_by_correct_count(self):
        frames=[SimpleNamespace(height=32,width=32,is_corrupt=False,pts=t,time_base=Fraction(1,24))
                for t in [0,1,2,2,4,5]]
        container=MagicMock()
        container.__enter__.return_value=container
        container.decode.side_effect=lambda _:iter(frames)
        with patch('scripts.finalize_gta_release.av.open',return_value=container):
            self.assertEqual(video_shape('duplicate.mp4')[0],6)
            with self.assertRaisesRegex(ValueError,'Non-CFR timestamp'):
                video_shape('duplicate.mp4',fps=24)

    def test_missing_time_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gap.mp4'
            write_clip(path,[0,1,2,4,5,6])
            with self.assertRaisesRegex(ValueError,'Non-CFR timestamp'):
                video_shape(path,fps=24)


if __name__=='__main__':unittest.main()
