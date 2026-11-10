import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib
from adapt_gta555_record_indices import adapt_episode


class IndexAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'adapted'
        self.output.mkdir()
        self.episode = {'episode_id': 'ep', 'path': 'data/episode', 'spools': {}}
        self.reports = {}
        for kind in ('world', 'agent_000'):
            raw = self.root / (kind + '.jsonl')
            idx = self.root / (kind + '.sqlite3')
            row = {'episode_id': 'ep', 'server_clock_domain': 'server', 'sample_seq': 5,
                   'client_clock_domain': 'client', 'client_sample_seq': 8}
            encoded = (json.dumps(row) + '\n').encode()
            raw.write_bytes(encoded)
            with closing(sqlite3.connect(idx)) as db, db:
                db.execute('CREATE TABLE records(ordinal,seq,domain,client_domain,offset,length)')
                db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)', (0, 5 if kind == 'world' else 8, 'server', 'client', 0, len(encoded)))
            self.episode['spools'][kind] = {'source': raw.name, 'index': idx.name}
            self.reports['ep', kind] = {'signature': ['', len(encoded)], 'sha256': hashlib.sha256(encoded).hexdigest(), 'records': 1}

    def tearDown(self):
        self.temp.cleanup()

    def test_client_domain_and_raw_pointer_preserved(self):
        result = adapt_episode(self.root, None, self.episode, self.reports, self.output)
        with closing(sqlite3.connect(result['index'])) as db:
            row = db.execute('SELECT domain,seq,data FROM camera').fetchone()
            self.assertEqual(row[:2], ('client', 8))
            pointer = json.loads(zlib.decompress(row[2]))
            self.assertEqual(pointer['offset'], 0)
            self.assertTrue(Path(pointer['_jsonl']).is_file())
            self.assertEqual(db.execute('SELECT domain,seq FROM world').fetchone(), ('server', 5))
            self.assertEqual(db.execute('SELECT count(*) FROM entity').fetchone()[0], 0)

    def test_same_length_changed_spool_rejected(self):
        path = self.root / 'world.jsonl'
        path.write_bytes(path.read_bytes().replace(b'server', b'xxxxxx'))
        with self.assertRaisesRegex(ValueError, 'changed spool'):
            adapt_episode(self.root, None, self.episode, self.reports, self.output)

    def test_legacy_estimated_sequence_matches_frozen_index(self):
        path = self.root / 'agent_000.jsonl'
        row = json.loads(path.read_text())
        row['sample_seq_estimate'] = row.pop('client_sample_seq')
        encoded = (json.dumps(row) + '\n').encode()
        path.write_bytes(encoded)
        report = self.reports['ep', 'agent_000']
        report['signature'][1] = len(encoded)
        report['sha256'] = hashlib.sha256(encoded).hexdigest()
        with closing(sqlite3.connect(self.root / 'agent_000.sqlite3')) as db, db:
            db.execute('UPDATE records SET length=?', (len(encoded),))
        result = adapt_episode(self.root, None, self.episode, self.reports, self.output)
        self.assertEqual(result['counts']['agent_000'], 1)

    def test_missing_source_not_marked_ready(self):
        (self.root / 'world.jsonl').unlink()
        self.assertIsNone(adapt_episode(self.root, None, self.episode, self.reports, self.output))
        self.assertFalse((self.output / 'ep.sqlite3').exists())


if __name__ == '__main__':
    unittest.main()
