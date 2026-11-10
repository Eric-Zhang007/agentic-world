from PIL import Image, ImageDraw

from compiler.projection import project_object


def draw_frame(frame, observations, matching, objects, camera, output):
    image = Image.open(frame["path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    matched = {m["obs_id"]: m for m in matching["matched"]}
    for obj in objects:
        if not obj.state.get("matchable", True):
            continue
        p = project_object(obj, camera)
        if not p or not p.get("visible") or not p.get("bbox"):
            continue
        x0, y0, x1, y1 = map(int, p["bbox"])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.width-1, x1), min(image.height-1, y1)
        for x in range(x0, x1, 10):
            draw.line((x, y0, min(x+5, x1), y0), fill="cyan", width=2)
            draw.line((x, y1, min(x+5, x1), y1), fill="cyan", width=2)
        for y in range(y0, y1, 10):
            draw.line((x0, y, x0, min(y+5, y1)), fill="cyan", width=2)
            draw.line((x1, y, x1, min(y+5, y1)), fill="cyan", width=2)
    label_regions = []
    for obj in observations:
        record = matched.get(obj.obs_id)
        color = "lime" if record else "red"
        box = obj.bbox_xyxy
        draw.rectangle(box, outline=color, width=2)
        label = f"{obj.category} {obj.track_id.split(':')[-1]} -> "
        label += f"{record['world_id']} {record['match_confidence']:.2f}" if record else "UNMATCHED"
        text_width = draw.textlength(label) + 6
        x, y = max(0, min(box[0], image.width-text_width)), max(0, box[1]-14)
        while any(abs(y-old_y) < 14 and x < old_x+old_width and x+text_width > old_x
                  for old_x, old_y, old_width in label_regions) and y+28 < image.height:
            y += 14
        label_regions.append((x, y, text_width))
        draw.rectangle((x, y, min(image.width, x+text_width), y+14), fill="black")
        draw.text((x, y), label, fill=color)
    image.save(output)
