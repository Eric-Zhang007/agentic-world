import base64
import io
import json
from collections import Counter

from PIL import Image

from agent.client import AgentClient


def plan_rollout(config, world, goal, recent_decisions=None):
    objects = world.query_world()
    summary = [{"object_id": o.object_id, "category": o.category, "provenance": o.provenance}
               for o in objects[:50]]
    message = AgentClient(config).complete([
        {"role": "system", "content":
         "Plan a short video rollout of the existing explicit world. Do not invent entities or edit "
         "world facts. Output only JSON with rollout_intention (string) and reason (string)."},
        {"role": "user", "content": json.dumps({
            "goal": goal, "world_summary": summary,
            "recent_decisions": recent_decisions or []})}])
    result = json.loads(message.get("content") or "")
    if not isinstance(result.get("rollout_intention"), str) or not result["rollout_intention"].strip():
        raise ValueError("Agent plan lacks rollout_intention")
    return result


def decide(config, world_db, intention, report, recent_decisions=None, observation_store=None, frames=None):
    objects = world_db.query_world()
    prompt = {
        "world_summary": {"object_count": len(objects), "categories": dict(Counter(o.category for o in objects)),
                          "tick": world_db.get_meta("tick", 0)},
        "rollout_intention": intention,
        "discrepancy_report": report,
        "recent_decisions": recent_decisions or [],
        "policy": [
            "Return one action: ACCEPT, RECONCILE_WORLD, or REGENERATE.",
            "An unmatched generated object is not an authoritative fact.",
            "Generated-only repetition is not independent evidence. RECONCILE_WORLD requires original/manual evidence IDs supporting the exact change.",
            "RECONCILE_WORLD edits use op=update, object_id, evidence_ids and changed fields.",
            "Choose REGENERATE for severe missing or spatial inconsistency.",
            "Conflicting labels on the same image region are perception ambiguity, not two objects.",
            "Missing objects also unverified on the compiled reference cannot be attributed solely to H3.",
            "A compiled reference is not independent original evidence. Do not claim semantic success when assessability is poor.",
            "Return strict JSON with action, reason, edits, confidence.",
        ],
    }
    content = [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]
    for frame in (frames or [])[:3]:
        image = Image.open(frame["path"]).convert("RGB")
        image.thumbnail((640, 360))
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG")
        content.append({"type": "image_url", "image_url": {"url":
            "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode()}})
    messages = [
        {"role": "system", "content": "You maintain explicit persistent world state. Never promote renderer hallucinations automatically."},
        {"role": "user", "content": content},
    ]
    definitions = [
        ("query_world", {"category": {"type": "string"}}, []),
        ("get_object", {"object_id": {"type": "string"}}, ["object_id"]),
        ("get_recent_observations", {}, []),
        ("get_track_history", {"track_id": {"type": "string"}}, ["track_id"]),
        ("get_evidence", {"evidence_ids": {"type": "array", "items": {"type": "string"}}}, ["evidence_ids"]),
    ]
    tools = [{"type": "function", "function": {
        "name": name, "description": "Query bounded relevant world or observation information.",
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}} for name, properties, required in definitions]
    trace = []
    for _ in range(config.get("max_tool_rounds", 6)):
        message = AgentClient(config).complete(messages, tools)
        messages.append(message)
        if not message.get("tool_calls"):
            break
        for call in message["tool_calls"]:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            if name == "query_world":
                result = [{"object_id": o.object_id, "category": o.category, "provenance": o.provenance}
                          for o in world_db.query_world(args.get("category"))[:50]]
            elif name == "get_object":
                obj = world_db.get_object(args["object_id"])
                result = obj.to_dict() if obj else None
                if result:
                    result.pop("appearance_embedding", None)
            elif name == "get_evidence":
                result = world_db.evidence(args["evidence_ids"])
            elif name == "get_recent_observations":
                result = observation_store.get_recent_objects() if observation_store else []
            elif name == "get_track_history":
                result = observation_store.get_track_history(args["track_id"]) if observation_store else []
            else:
                result = {"error": "Unknown tool"}
            trace.append({"tool": name, "arguments": args})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    text = message.get("content") or ""
    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("Decision is not an object")
    except ValueError:
        result = {"action": "REGENERATE", "reason": "Agent returned non-JSON",
                  "edits": [], "confidence": 0.0, "raw": text}
    if result.get("action") not in {"ACCEPT", "RECONCILE_WORLD", "REGENERATE"}:
        result["action"] = "REGENERATE"
    result["tool_trace"] = trace
    return result
