"""Explicit mutation/query tools; generated observations cannot authorize mutations."""
from compiler.projection import get_visible_objects
from state.models import WorldObject
from verifier.reconcile import apply_reconciliation


class WorldTools:
    def __init__(self, world, observations, renderer=None):
        self.world = world
        self.observations = observations
        self.renderer = renderer

    def query_world(self, category=None):
        return [o.to_dict() for o in self.world.query_world(category)]

    def get_object(self, object_id):
        obj = self.world.get_object(object_id)
        return obj.to_dict() if obj else None

    def get_visible_objects(self):
        return [o.to_dict() for o in get_visible_objects(self.world.query_world(), self.world.get_meta("camera"))]

    def get_recent_observations(self, limit=30):
        return self.observations.get_recent_objects(limit)

    def update_object(self, object_id, changes, evidence_ids, reason):
        return apply_reconciliation(self.world, {"action": "RECONCILE_WORLD",
            "edits": [{"op": "update", "object_id": object_id, "reason": reason,
                       "evidence_ids": evidence_ids, **changes}]})

    def move_object(self, object_id, position, evidence_ids, reason):
        return self.update_object(object_id, {"position": position}, evidence_ids, reason)

    def set_attribute(self, object_id, name, value, evidence_ids, reason):
        obj = self.world.get_object(object_id)
        if obj is None:
            raise KeyError(object_id)
        return self.update_object(object_id, {"attributes": {**obj.attributes, name: value}}, evidence_ids, reason)

    def spawn_object(self, obj, evidence_ids, reason):
        if self.world.get_object(obj["object_id"]):
            raise ValueError("Object already exists")
        records = self.world.evidence(evidence_ids)
        if not any(r["object_id"] == obj["object_id"] and r["payload"].get("spawn") == obj for r in records):
            raise ValueError("Independent original/manual evidence must authorize exact spawn")
        self.world.save(WorldObject(**obj), reason, evidence_ids)

    def remove_object(self, object_id, evidence_ids, reason):
        records = self.world.evidence(evidence_ids)
        if not any(r["object_id"] == object_id and r["payload"].get("remove") is True for r in records):
            raise ValueError("Independent evidence must authorize removal; generated absence is insufficient")
        self.world.remove(object_id, reason, evidence_ids)

    def render_rollout(self, **request):
        if self.renderer is None:
            raise ValueError("No renderer configured")
        return str(self.renderer.render(**request))

    def accept_rollout(self, rollout_id):
        self.world.set_meta("last_accepted_rollout", rollout_id)

    def reroll(self, **request):
        return self.render_rollout(**request)

    def inspect_rollout(self, video, output_dir, perception_config):
        from pathlib import Path
        from perception.frame_sampler import sample_frames
        from perception.detector import PerceptionPipeline
        output = Path(output_dir)
        frames = sample_frames(video, output / "sampled_frames",
                               perception_config["frame_interval_sec"])
        pipeline = PerceptionPipeline(perception_config)
        return [{"frame": frame, "objects": pipeline.detect_frame(
            frame, output / "segmentation" / str(frame["frame_id"]), output.name)} for frame in frames]

    def match_rollout(self, frame_observations, matching_config):
        from matching.matcher import match
        identities = {}
        return [{"frame_id": item["frame"]["frame_id"], "matching": match(
            self.world.query_world(), item["objects"], self.world.get_meta("camera"),
            matching_config, identities).to_dict()} for item in frame_observations]

    def get_discrepancy_report(self, rollout_id, frame_matches, matching_config):
        from verifier.discrepancy import discrepancy
        return discrepancy(rollout_id, frame_matches, matching_config)
