from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from common import save_json
from compiler.projection import project_object


def compile_world(objects, camera, output_dir, intention):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = camera["width"], camera["height"]
    if camera.get("whitebox_source_root"):
        return compile_inverse(objects, camera, output_dir, intention)
    image = Image.new("RGB", (width, height), (170, 208, 230))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height // 2, width, height), fill=(96, 145, 86))
    semantic = np.zeros((height, width), np.uint16)
    depth = np.full((height, width), np.nan, np.float32)
    projections = {o.object_id: project_object(o, camera) for o in objects}
    colors = {"road": (106, 110, 112), "car": (210, 47, 44), "person": (230, 180, 125),
              "tree": (44, 112, 63)}
    labels = {}
    for i, obj in sorted(enumerate(objects, 1), key=lambda p: -(p[1].position or [0, 0, 100])[2]):
        projection = projections[obj.object_id]
        if not projection or not projection["visible"] or not projection["bbox"]:
            continue
        x0, y0, x1, y1 = projection["bbox"]
        x0, x1 = np.clip([int(x0), int(x1)], 0, width)
        y0, y1 = np.clip([int(y0), int(y1)], 0, height)
        color = colors.get(obj.category, (155, 155, 155))
        draw.rectangle((x0, y0, x1, y1), fill=color)
        if obj.category == "car":
            draw.rectangle((x0 + 8, y0 + 5, x1 - 8, y0 + max(8, (y1-y0)//3)), fill=(175, 217, 228))
            r = max(3, (y1-y0)//7)
            for x in (x0 + r*2, x1 - r*2):
                draw.ellipse((x-r, y1-r*2, x+r, y1), fill=(35, 35, 35))
        elif obj.category == "tree":
            draw.rectangle((int((x0+x1)/2)-5, y0, int((x0+x1)/2)+5, y1), fill=(105, 88, 68))
            draw.ellipse((x0, y0, x1, int(y0+(y1-y0)*0.75)), fill=color)
        elif obj.category == "person":
            draw.rectangle((x0, y0, x1, y1), fill=(170, 208, 230))
            draw.ellipse((x0, y0, x1, y0+(x1-x0)), fill=color)
            draw.rectangle((x0+2, y0+(x1-x0), x1-2, y1-15), fill=(233, 190, 35))
            draw.line((x0+3, y1-15, x0+3, y1), fill=(38, 42, 46), width=5)
            draw.line((x1-3, y1-15, x1-3, y1), fill=(38, 42, 46), width=5)
        semantic[y0:y1, x0:x1] = i
        depth[y0:y1, x0:x1] = projection["depth"]
        labels[str(i)] = {"object_id": obj.object_id, "category": obj.category}
    image.save(output_dir / "reference.png")
    Image.fromarray(semantic).save(output_dir / "semantic.png")
    np.save(output_dir / "depth.npy", depth)
    save_json(output_dir / "projections.json", projections)
    save_json(output_dir / "semantic_ids.json", labels)
    description = [{"id": o.object_id, "category": o.category,
                    "position": o.position, "attributes": o.attributes} for o in objects]
    prompt = ("Use Image 1 as the scene layout reference. Static camera, one continuous shot. "
              "Preserve object counts and their relative positions. Realistic daylight. "
              "Reproduce the pictured scene faithfully; do not add foreground entities. "
              + intention + "\nWorld entities: " + str(description))
    return {"prompt": prompt, "first_frame": str(output_dir / "reference.png"),
            "projections": projections,
        "unsupported_conditions": ["depth.npy (box proxy)", "semantic.png (box proxy)"]}


def compile_inverse(objects, camera, output_dir, intention):
    """Reuse the supplied CPU mesh renderer, materializing the current working state."""
    sys.path.insert(0, camera["whitebox_source_root"])
    from whitebox.render import render_frame
    entities, characters = [], []
    for obj in objects:
        if obj.attributes.get("source_group") == "characters":
            joints = np.asarray(obj.attributes["joints"])
            original_center = (joints.min(0)+joints.max(0))/2
            joints += np.asarray(obj.position)-original_center
            characters.append({"id": obj.object_id, "radius": obj.attributes["radius"],
                               "keyframes": [{"time": 0, "joints": joints.tolist()}]})
        elif obj.attributes.get("shape"):
            entities.append({"id": obj.object_id, "shape": obj.attributes["shape"],
                             "keyframes": [{"time": 0, "position": obj.position,
                                            "rotation": np.degrees(obj.rotation or [0, 0, 0]).tolist(),
                                            "scale": obj.scale}]})
    scene = {"objects": entities, "characters": characters,
             "camera": {"projection": camera["projection"], "ortho_scale": camera["ortho_scale"],
                        "fov_degrees": camera["fov_degrees"], "keyframes": [
                            {"time": 0, "position": camera["position"], "target": camera["target"]}]}}
    rgb = render_frame(scene, 0, camera["width"], camera["height"])
    Image.fromarray(rgb).save(output_dir / "reference.png")
    save_json(output_dir / "materialized_scene.json", scene)
    projections = {o.object_id: project_object(o, camera) for o in objects}
    save_json(output_dir / "projections.json", projections)
    return {"prompt": "Preserve the geometry and spatial arrangement in Image 1. Static camera. "
                      + intention, "first_frame": str(output_dir / "reference.png"),
            "projections": projections,
            "unsupported_conditions": ["No depth/semantic conditioning in deployed Ref2VA"],
            "source": "reused supplied whitebox.render.render_frame"}
