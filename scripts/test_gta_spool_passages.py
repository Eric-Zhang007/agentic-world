import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import orjson
from scripts.prepare_gta_spool_passages import work
from memory.gta_release_media import source_media_path,rgb_file_index


class SourcePreparation(unittest.TestCase):
    def test_segment_local_index_preserves_global_source_metadata(self):
        row={'rgb_decode_index':34414,'rgb_segment_frame_index':0}
        self.assertEqual(rgb_file_index(row),0)
        self.assertEqual(row['rgb_decode_index'],34414)
        self.assertEqual(rgb_file_index({'rgb_decode_index':120}),120)
        for invalid in (-1,None,True):
            with self.assertRaises(ValueError):
                rgb_file_index({'rgb_decode_index':7,'rgb_segment_frame_index':invalid})

    def test_external_proxy_cache_preserves_rgb_root(self):
        record={'episode_root':'source','media_paths':{'proxy_video':'cache/proxy.mp4'}}
        manifest={'rgb_video':'rgb.mkv','proxy_video':'missing/proxy.mp4'}
        self.assertEqual(source_media_path(record,manifest,'proxy_video'),Path('cache/proxy.mp4'))
        self.assertEqual(source_media_path(record,manifest,'rgb_video'),Path('source/rgb.mkv'))

    def test_unaligned_frames_not_promoted_to_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();out=root/'out';out.mkdir()
            video=source/'video.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),24,(64,48))
            for _ in range(4):writer.write(np.zeros((48,64,3),np.uint8))
            writer.release()
            (source/'manifest.json').write_bytes(orjson.dumps({'resolution':[64,48],'rgb_video':'video.mp4','proxy_video':'video.mp4'}))
            (source/'frames.jsonl').write_bytes(b''.join(orjson.dumps({'proxy_frame_index':i,'rgb_decode_index':i,'status':'missing_final_alignment'})+b'\n' for i in range(4)))
            row={'episode_id':'e','agent_id':'a','segment':'s','source_manifest':str(source/'manifest.json'),'episode_root':str(source),'record_index':'unused'}
            with patch('scripts.prepare_gta_spool_passages.Records') as mock:
                mock.return_value.db.execute.return_value.fetchall.return_value=[]
                result=work((row,str(root),str(out)))
            self.assertEqual(result['decoded_frames'],{'rgb_video':4,'proxy_video':4})
            self.assertEqual(result['indexed_points'],0)
            self.assertEqual(result['invalid_frames'],4)
            self.assertEqual(work((row,str(root),str(out))),result)


if __name__=='__main__':unittest.main()
