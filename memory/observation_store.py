import json
from pathlib import Path
import sqlite3


class ObservationStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS frames(
            episode_id TEXT, attempt INTEGER, frame_id INTEGER, timestamp REAL, path TEXT,
            PRIMARY KEY(episode_id, attempt, frame_id))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS observations(
            episode_id TEXT, attempt INTEGER, obs_id TEXT PRIMARY KEY, frame_id INTEGER,
            track_id TEXT, category TEXT, bbox_json TEXT, confidence REAL, source TEXT,
            embedding_path TEXT, depth REAL, world_position_json TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS reports(
            episode_id TEXT, attempt INTEGER, report_json TEXT, PRIMARY KEY(episode_id, attempt))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS decisions(
            episode_id TEXT, attempt INTEGER, decision_json TEXT, PRIMARY KEY(episode_id, attempt))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS artifacts(
            obs_id TEXT PRIMARY KEY, mask_path TEXT, embedding_json TEXT)""")
        self.db.commit()

    def add_frame(self, episode_id, attempt, frame):
        self.db.execute("INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?)",
                        (episode_id, attempt, frame["frame_id"], frame["timestamp"], frame["path"]))

    def add_object(self, episode_id, attempt, obj):
        self.db.execute("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (episode_id, attempt, obj.obs_id, obj.frame_id, obj.track_id, obj.category,
                         json.dumps(obj.bbox_xyxy), obj.confidence, obj.source, None, obj.depth,
                         json.dumps(obj.world_position) if obj.world_position else None))
        self.db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?)",
                        (obj.obs_id, obj.mask, json.dumps(obj.visual_embedding)))

    def add_decision(self, episode_id, attempt, decision):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?)",
                            (episode_id, attempt, json.dumps(decision)))

    def get_recent_objects(self, limit=30):
        rows = self.db.execute(
            "SELECT obs_id,track_id,category,bbox_json,confidence,source FROM observations ORDER BY rowid DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(zip(("obs_id", "track_id", "category", "bbox", "confidence", "source"), row)) for row in rows]

    def get_track_history(self, track_id):
        rows = self.db.execute(
            "SELECT obs_id,frame_id,bbox_json,confidence FROM observations WHERE track_id=? ORDER BY frame_id",
            (track_id,)).fetchall()
        return [dict(zip(("obs_id", "frame_id", "bbox", "confidence"), row)) for row in rows]

    def get_recent_decisions(self, limit=3):
        rows = self.db.execute("SELECT decision_json FROM decisions ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [{"action": d.get("action"), "reason": d.get("reason"),
                 "applied_edits": d.get("applied_edits", [])}
                for row in reversed(rows) for d in [json.loads(row[0])]]

    def add_report(self, episode_id, attempt, report):
        self.db.execute("INSERT OR REPLACE INTO reports VALUES (?,?,?)",
                        (episode_id, attempt, json.dumps(report)))
        self.db.commit()
