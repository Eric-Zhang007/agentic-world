"""Exercise interruption recovery and rejection of stale delivery contents."""
from pathlib import Path
import tempfile
import unittest
import zipfile
import json
from unittest.mock import patch

from scripts.finalize_gta_release import write_archive
from scripts import finalize_gta_release
from scripts.complete_gta_release import copy_archive


class ArchiveTests(unittest.TestCase):
    def test_failed_review_blocks_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('qualification_summary.json','visual_review_complete.json','export_complete.json'):
                (root/name).write_text('{}')
            review=root/'visual_review'/'candidate';review.mkdir(parents=True)
            (review/'decision.json').write_text(json.dumps({'status':'REVIEW_FAILED'}))
            with patch('sys.argv',['finalize','--release',str(root)]):
                with self.assertRaisesRegex(ValueError,'Unfinished visual review'):
                    finalize_gta_release.main()
            self.assertFalse((root/'DELIVERY.json').exists())

    def test_missing_stage_blocks_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('sys.argv',['finalize','--release',directory]):
                with self.assertRaisesRegex(ValueError,'Processing stage incomplete'):
                    finalize_gta_release.main()

    def test_copy_restart_and_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.zip';target=root/'copied.zip'
            with zipfile.ZipFile(source,'w') as archive:archive.writestr('data','old')
            copy_archive(source,target);stamp=target.stat().st_mtime_ns
            copy_archive(source,target)
            self.assertEqual(stamp,target.stat().st_mtime_ns)
            with zipfile.ZipFile(source,'w') as archive:archive.writestr('data','new')
            with self.assertRaisesRegex(ValueError,'differs'):copy_archive(source,target)

    def test_partial_recovery_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'video.mp4';target=root/'episode.zip'
            source.write_bytes(b'video contents')
            target.with_name(target.name+'.partial').write_bytes(b'interrupted')
            write_archive(target,[(source,'samples/video.mp4')],{'index.json':'{}'})
            first=target.stat().st_mtime_ns
            write_archive(target,[(source,'samples/video.mp4')],{'index.json':'{}'})
            self.assertEqual(first,target.stat().st_mtime_ns)
            with zipfile.ZipFile(target) as archive:self.assertIsNone(archive.testzip())

    def test_same_size_changed_media_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'video.mp4';target=root/'episode.zip'
            source.write_bytes(b'old');write_archive(target,[(source,'video.mp4')])
            source.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError,'Stale archive content'):
                write_archive(target,[(source,'video.mp4')])

    def test_missing_members_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'video.mp4';target=root/'episode.zip'
            source.write_bytes(b'video');write_archive(target,[(source,'video.mp4')])
            with self.assertRaisesRegex(ValueError,'does not match'):
                write_archive(target,[(source,'video.mp4')],{'index.json':'{}'})


if __name__=='__main__':unittest.main()
