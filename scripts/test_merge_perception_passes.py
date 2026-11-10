import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from scripts.merge_perception_passes import merge


class MergeTests(unittest.TestCase):
    def make(self, root, name, color):
        folder = root/name
        folder.mkdir()
        Image.new('RGB', (32, 24), color).save(folder/'input.png')
        Image.new('L', (32, 24), 255).save(folder/'mask.png')
        (folder/'embedding.npy').write_bytes(b'test path only')
        record = {'image_size': [32, 24], 'observations': [{'observation_id': 'same_id',
                  'mask_path': 'mask.png', 'embedding_path': 'embedding.npy', 'category': 'building'}]}
        path = folder/'result.json'
        path.write_text(json.dumps(record))
        return path

    def test_merge_preserves_sources_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self.make(root, n, 'red') for n in ('a', 'b')]
            before = [p.read_bytes() for p in paths]
            result = json.loads(merge(paths, root/'merged').read_text())
            items = result['observations']
            self.assertEqual(len({r['observation_id'] for r in items}), 2)
            self.assertEqual([p.read_bytes() for p in paths], before)
            self.assertTrue(all(Path(r['mask_path']).is_absolute() for r in items))

    def test_wrong_rgb_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self.make(root, 'a', 'red'), self.make(root, 'b', 'blue')]
            with self.assertRaisesRegex(ValueError, 'identical'):
                merge(paths, root/'merged')
            self.assertFalse((root/'merged').exists())


if __name__ == '__main__':
    unittest.main()
