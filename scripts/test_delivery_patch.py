import json
import tempfile
import unittest
from pathlib import Path
from apply_delivery_patch import apply

class PatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.root=self.base/'dataset';self.package=self.base/'patch'
        original=[{'sample_id':'a'},{'sample_id':'b'}]
        corrected=[original[0]]
        self.original=original
        self.names=['manifests/train.jsonl','manifests/train_round_00.jsonl']
        for name in self.names:
            for path,data in ((self.root/name,original),(self.package/'payload'/name,corrected)):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(''.join(json.dumps(r)+'\n' for r in data))
        sample=self.root/'samples/a'; sample.mkdir(parents=True)
        (sample/'sample.json').write_text(json.dumps({'fps':24,'frame_count':120}))
        (sample/'target_rgb.mp4').write_bytes(b'original-video')
        dest=self.package/'payload/samples/a/target_rgb.mp4';dest.parent.mkdir(parents=True)
        dest.write_bytes(b'repaired-video')
        self.spec={'repair_ids':['a'],'repair_records':1,'exclude_records':1,'remaining':1,'original_manifest_records':{name:original for name in self.names},'files':[{'path':p.relative_to(self.package/'payload').as_posix(),'bytes':p.stat().st_size} for p in (self.package/'payload').rglob('*') if p.is_file()]}
        (self.package/'patch.json').write_text(json.dumps(self.spec))

    def test_apply_backup_rounds_and_repeat(self):
        self.assertEqual(apply(self.root,self.package)['status'],'check_passed')
        self.assertFalse((self.root/'patch_backup').exists())
        self.assertEqual(apply(self.root,self.package,True)['status'],'applied')
        for name in self.names:
            self.assertNotIn('"b"',(self.root/name).read_text())
            self.assertIn('"b"',(self.root/'patch_backup'/name).read_text())
        self.assertEqual((self.root/'patch_backup/samples/a/target_rgb.mp4').read_bytes(),b'original-video')
        self.assertEqual(apply(self.root,self.package,True)['status'],'already_applied')

    def test_wrong_dataset_rejected_before_writing(self):
        (self.root/self.names[0]).write_text('{"sample_id":"other"}\n')
        with self.assertRaises(ValueError):apply(self.root,self.package,True)
        self.assertFalse((self.root/'patch_backup').exists())

    def test_partial_application_resumes(self):
        apply(self.root,self.package,True)
        (self.root/'GTA_MEMORY_PATCH_APPLIED.json').unlink()
        (self.root/self.names[1]).write_text((self.root/'patch_backup'/self.names[1]).read_text())
        self.assertEqual(apply(self.root,self.package,True)['status'],'applied')
        self.assertIn('"b"',(self.root/'patch_backup'/self.names[0]).read_text())

    def test_incomplete_payload_rejected(self):
        (self.package/'payload/samples/a/target_rgb.mp4').write_bytes(b'bad')
        with self.assertRaises(ValueError):apply(self.root,self.package,True)
        self.assertFalse((self.root/'patch_backup').exists())

if __name__=='__main__':unittest.main()
