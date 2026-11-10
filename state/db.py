import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from state.models import WorldObject


class WorldDB:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS objects(
            object_id TEXT PRIMARY KEY, category TEXT NOT NULL,
            position TEXT, rotation TEXT, scale TEXT, bbox_3d TEXT,
            appearance_embedding TEXT, attributes TEXT NOT NULL,
            state TEXT NOT NULL, confidence REAL, provenance TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS world_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS evidence(
            evidence_id TEXT PRIMARY KEY, source TEXT NOT NULL,
            object_id TEXT, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS edits(
            edit_id INTEGER PRIMARY KEY, object_id TEXT NOT NULL,
            before_json TEXT, after_json TEXT, reason TEXT NOT NULL,
            evidence_ids TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)

    def query_world(self, category=None):
        rows = self.connection.execute("SELECT * FROM objects WHERE provenance != 'RENDER_ONLY' ORDER BY object_id")
        objects = [self._decode(row) for row in rows]
        return [obj for obj in objects if category is None or obj.category == category]

    @staticmethod
    def _decode(row):
        value = dict(row)
        for key in ("position", "rotation", "scale", "bbox_3d", "appearance_embedding", "attributes", "state"):
            value[key] = json.loads(value[key]) if value[key] is not None else None
        return WorldObject(**value)

    def get_object(self, object_id):
        row = self.connection.execute("SELECT * FROM objects WHERE object_id=?", (object_id,)).fetchone()
        return self._decode(row) if row else None

    def get_meta(self, key, default=None):
        row = self.connection.execute("SELECT value FROM world_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    @contextmanager
    def transaction(self):
        outer = self._transaction_depth == 0
        self._transaction_depth += 1
        try:
            yield
            if outer:
                self.connection.commit()
        except BaseException:
            if outer:
                self.connection.rollback()
            raise
        finally:
            self._transaction_depth -= 1

    def set_meta(self, key, value):
        with self.transaction():
            self.connection.execute("INSERT OR REPLACE INTO world_meta VALUES (?,?)", (key, json.dumps(value)))

    def save(self, obj, reason, evidence_ids=()):
        if obj.provenance not in {"OBSERVED", "INFERRED", "AGENT_CREATED", "RENDER_ONLY", "UNKNOWN"}:
            raise ValueError("Invalid provenance")
        before = self.get_object(obj.object_id)
        data = obj.to_dict()
        values = [json.dumps(value) if key in {
            "position", "rotation", "scale", "bbox_3d", "appearance_embedding", "attributes", "state"
        } else value for key, value in data.items()]
        with self.transaction():
            self.connection.execute(
                "INSERT OR REPLACE INTO objects VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
            self.connection.execute(
                "INSERT INTO edits(object_id,before_json,after_json,reason,evidence_ids) VALUES (?,?,?,?,?)",
                (obj.object_id, json.dumps(before.to_dict()) if before else None,
                 json.dumps(data), reason, json.dumps(list(evidence_ids))))

    def remove(self, object_id, reason, evidence_ids=()):
        before = self.get_object(object_id)
        if before is None:
            raise KeyError(object_id)
        with self.transaction():
            self.connection.execute("DELETE FROM objects WHERE object_id=?", (object_id,))
            self.connection.execute(
                "INSERT INTO edits(object_id,before_json,after_json,reason,evidence_ids) VALUES (?,?,?,?,?)",
                (object_id, json.dumps(before.to_dict()), None, reason, json.dumps(list(evidence_ids))))

    def snapshot(self):
        return {"objects": [o.to_dict() for o in self.query_world()],
                "camera": self.get_meta("camera"), "tick": self.get_meta("tick", 0)}

    def add_evidence(self, evidence_id, source, object_id, payload):
        if source not in {"ORIGINAL", "MANUAL"}:
            raise ValueError("Generated frames are observations, not independent evidence")
        with self.transaction():
            self.connection.execute("INSERT INTO evidence VALUES (?,?,?,?)",
                                    (evidence_id, source, object_id, json.dumps(payload)))

    def evidence(self, ids):
        result = []
        for key in ids:
            row = self.connection.execute("SELECT * FROM evidence WHERE evidence_id=?", (key,)).fetchone()
            if row:
                value = dict(row)
                value["payload"] = json.loads(value["payload"])
                result.append(value)
        return result
