def apply_reconciliation(world_db, decision, evidence_ids=()):
    if decision.get("action") != "RECONCILE_WORLD":
        return []
    pending = []
    for edit in decision.get("edits", []):
        ids = edit.get("evidence_ids", list(evidence_ids))
        records = world_db.evidence(ids)
        if not records or any(record["object_id"] != edit.get("object_id") for record in records):
            raise ValueError("Independent object-specific original/manual evidence required")
        if edit.get("op") != "update":
            raise ValueError("Unsupported operation; use explicit WorldTools for spawn/remove")
        obj = world_db.get_object(edit["object_id"])
        if obj is None:
            raise ValueError("Cannot update missing object")
        for key in ("position", "rotation", "scale", "attributes", "state", "confidence", "provenance"):
            if key in edit:
                if not any(key in record["payload"].get("allowed_changes", {})
                           and record["payload"]["allowed_changes"][key] == edit[key] for record in records):
                    raise ValueError(f"Independent evidence does not support {key}")
                setattr(obj, key, edit[key])
        pending.append((obj, edit.get("reason", "agent reconciliation"), ids))
    # Validate the entire proposal before writes, including independent evidence.
    for obj, reason, ids in pending:
        world_db.save(obj, reason, ids)
    return [obj.object_id for obj, _, _ in pending]
