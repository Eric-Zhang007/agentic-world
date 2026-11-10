"""Appearance references, never world facts or independent evidence."""

import base64
from io import BytesIO
import math
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError


class AppearanceMemory:
    """Append-only reference metadata in WorldDB; image files are derived assets.

    Bounding boxes are normalized (left, top, right, bottom). Returned images
    are detached PIL images, cropped when a box is supplied. For an in-memory
    WorldDB, PNG bytes live in metadata instead of a separate database.
    """

    META_KEY = "appearance_reference_memory"

    def __init__(self, world):
        self.world = world
        rows = world.connection.execute("PRAGMA database_list").fetchall()
        path = next((row[2] for row in rows if row[1] == "main"), "")
        self.directory = Path(path).parent / "reference_memory" if path else None

    @staticmethod
    def _label(value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
        return value

    @staticmethod
    def _number(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def add(self, object_id, image_path, prompt, view="unknown", state="default",
            source="AGENT_CREATED", accepted=False, original=False,
            bbox_normalized=None, quality=1.0, timestamp=0.0, evidence=None):
        object_id = self._label(object_id, "object_id")
        view = self._label(view, "view")
        state = self._label(state, "state")
        source = self._label(source, "source")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(accepted, bool) or not isinstance(original, bool):
            raise ValueError("accepted and original must be booleans")
        quality = self._number(quality, "quality")
        timestamp = self._number(timestamp, "timestamp")
        if quality < 0:
            raise ValueError("quality must be nonnegative")
        bbox = None
        if bbox_normalized is not None:
            if not isinstance(bbox_normalized, (tuple, list)) or len(bbox_normalized) != 4:
                raise ValueError("bbox_normalized must contain four coordinates")
            bbox = [self._number(v, "bbox coordinate") for v in bbox_normalized]
            left, top, right, bottom = bbox
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("bbox_normalized must have positive area within [0, 1]")
        path = Path(image_path).resolve()
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.width < 1 or image.height < 1:
                    raise ValueError("image must have positive dimensions")
                frame = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Invalid reference image: {path}") from exc
        buffer = BytesIO()
        frame.save(buffer, format="PNG")
        ref = {
            "id": uuid4().hex, "object_id": object_id, "image_path": str(path),
            "prompt": prompt, "view": view, "state": state, "source": source,
            "accepted": accepted, "original": original, "bbox_normalized": bbox,
            "quality": quality, "timestamp": timestamp, "size": list(frame.size),
        }
        if evidence is not None:
            ref['evidence'] = evidence
        # Never reuse caller filenames: later renders cannot overwrite identity.
        asset = None
        created = False
        try:
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                asset = self.directory / (ref["id"] + ".png")
                with asset.open("xb") as output:
                    created = True
                    output.write(buffer.getvalue())
                ref["reference_path"] = str(asset)
            else:
                ref["png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
            with self.world.transaction():
                records = self.world.get_meta(self.META_KEY, [])
                records.append(ref)
                self.world.set_meta(self.META_KEY, records)
        except BaseException:
            if created:asset.unlink(missing_ok=True)
            raise
        return dict(ref)

    def identity_anchor(self, object_id):
        """Stable text provenance, independent of the current image budget."""
        self._label(object_id, 'object_id')
        rows = [r for r in self.world.get_meta(self.META_KEY, [])
                if r['object_id'] == object_id and
                (r['original'] or (r['accepted'] and r['state'] == 'identity'))]
        originals = [r for r in rows if r['original']]
        candidates = originals or rows
        if not candidates:
            return None
        ref = min(candidates, key=lambda r: (r['timestamp'], r['id']))
        return {key: ref[key] for key in
                ('id', 'object_id', 'prompt', 'view', 'source', 'original', 'accepted', 'timestamp')}

    def select(self, object_id, view="unknown", state="default", limit=2):
        """Prefer exact state/view; reserve a slot for original identity.

        An original in another state is explicitly identity-only. It must not
        transfer its state to the requested depiction. Unknown is a view label,
        not a wildcard state. All non-original references require acceptance.
        """
        self._label(object_id, "object_id")
        self._label(view, "view")
        self._label(state, "state")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        if not limit:
            return []
        refs = [r for r in self.world.get_meta(self.META_KEY, [])
                if r["object_id"] == object_id and (r["original"] or r["accepted"])]

        def rank(ref):
            view_score = (0 if ref["view"] == "unknown" else
                          2 if ref["view"] == view else 1)
            return (view_score, ref["quality"], ref["timestamp"], ref["id"])

        matching = sorted((r for r in refs if r["state"] == state), key=rank, reverse=True)
        originals = sorted((r for r in refs if r["original"]), key=rank, reverse=True)
        if not originals and state == 'identity' and matching:
            # Keep an early accepted identity anchor instead of recursively
            # replacing every reference with the newest generated appearance.
            originals = [min(matching, key=lambda r: (r['timestamp'],r['id']))]
        chosen = matching[:1]
        if originals and originals[0] not in chosen and not any(r["original"] for r in chosen) and len(chosen) < limit:
            chosen.append(originals[0])
        for ref in matching:
            if len(chosen) >= limit:
                break
            if ref not in chosen:
                chosen.append(ref)
        return [dict(r, role="state_reference" if r["state"] == state else "identity_only")
                for r in chosen]

    def _image(self, ref):
        source = (BytesIO(base64.b64decode(ref["png_base64"])) if "png_base64" in ref
                  else ref["reference_path"])
        with Image.open(source) as image:
            image.load()
            bbox = ref["bbox_normalized"]
            if bbox is None:
                return image.copy()
            width, height = image.size
            pixels = (math.floor(bbox[0] * width), math.floor(bbox[1] * height),
                      math.ceil(bbox[2] * width), math.ceil(bbox[3] * height))
            return image.crop(pixels)

    def prompt_and_images(self, requests):
        """Return aligned prompts, PIL images and refs, in request order."""
        refs, images, lines = [], [], []
        for request in requests:
            object_id = request["object_id"]
            view, state = request.get("view", "unknown"), request.get("state", "default")
            selected = self.select(object_id, view, state)
            lines.append(f"Object {object_id}; requested view={view}; requested state={state}.")
            if not selected:
                lines.append("No accepted appearance reference available.")
            for ref in selected:
                images.append(self._image(ref))
                refs.append(ref)
                role = ("Identity only; do not copy reference state" if ref["role"] == "identity_only"
                        else "Matching-state appearance reference")
                lines.append(f"Image {len(images)}: {role}; reference view={ref['view']}; "
                             f"reference state={ref['state']}; reference prompt: {ref['prompt']}")
        return {"prompt": "\n".join(lines), "images": images, "refs": refs}
