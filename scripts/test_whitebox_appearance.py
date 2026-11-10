import tempfile
from pathlib import Path
import unittest

from PIL import Image

from state.db import WorldDB
from state.models import WorldObject
from whitebox.appearance import AppearanceMemory


class AppearanceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.world = WorldDB(self.root / "world.sqlite")
        self.addCleanup(self.world.connection.close)
        self.memory = AppearanceMemory(self.world)
        self.path = self.root / "frame.png"
        image = Image.new("RGB", (20, 10), "red")
        image.paste("blue", (10, 0, 20, 10))
        image.save(self.path)

    def add(self, **kwargs):
        return self.memory.add("door", self.path, "A painted door", **kwargs)

    def test_real_crop_and_original_survive_source_overwrite(self):
        original = self.add(original=True, source="GENERATED_SEED", state="closed",
                            bbox_normalized=[0.5, 0, 1, 1])
        Image.new("RGB", (20, 10), "green").save(self.path)
        current = self.add(accepted=True, state="open")
        result = self.memory.prompt_and_images([{"object_id": "door", "state": "open"}])
        self.assertEqual([r["id"] for r in result["refs"]], [current["id"], original["id"]])
        self.assertEqual(result["images"][1].size, (10, 10))
        self.assertEqual(result["images"][1].getpixel((0, 0)), (0, 0, 255))
        self.assertIn("Identity only; do not copy reference state", result["prompt"])
        self.assertIn("A painted door", result["prompt"])
        stored = self.world.get_meta(self.memory.META_KEY)
        self.assertEqual(stored[0], original)
        self.assertEqual(Path(original["reference_path"]).parent, self.root / "reference_memory")

    def test_state_view_and_acceptance(self):
        self.add(source="GENERATED", view="front", state="open", quality=100)
        self.add(accepted=True, view="front", state="closed", quality=100)
        self.add(accepted=True, view="unknown", state="open", quality=100)
        side = self.add(accepted=True, view="side", state="open")
        front = self.add(accepted=True, view="front", state="open")
        self.assertEqual([r["id"] for r in self.memory.select("door", "front", "open")],
                         [front["id"], side["id"]])
        self.assertEqual(self.memory.select("door", "side", "open", 1)[0]["id"], side["id"])
        self.assertEqual(self.memory.select("door", state="broken"), [])
        self.assertEqual(self.memory.select("other"), [])
        self.assertEqual(self.memory.select("door", limit=0), [])

    def test_metadata_only_and_reopen(self):
        self.world.save(WorldObject("door", "door", state={"open": False}), "fixture")
        before = self.world.snapshot()
        edits = self.world.connection.execute("SELECT * FROM edits").fetchall()
        ref = self.add(original=True, source="GENERATED_SEED")
        self.assertEqual(self.world.snapshot(), before)
        self.assertEqual(self.world.connection.execute("SELECT * FROM edits").fetchall(), edits)
        self.assertEqual(self.world.connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], 0)
        reopened = WorldDB(self.root / "world.sqlite")
        try:
            self.assertEqual(AppearanceMemory(reopened).select("door")[0]["id"], ref["id"])
        finally:
            reopened.connection.close()

    def test_invalid_inputs_leave_no_metadata(self):
        for bbox in ([0, 0, float("nan"), 1], [0, 0, float("inf"), 1],
                     [0, 0, 0, 1], [-1, 0, 1, 1], [0, 0, 2, 1], [0, 1, 1, 0], [0]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                self.add(bbox_normalized=bbox)
        for kwargs in ({"quality": float("nan")}, {"timestamp": float("inf")},
                       {"quality": -1}, {"accepted": "yes"}, {"state": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.add(**kwargs)
        with self.assertRaises(ValueError):
            self.memory.add("door", self.root / "missing.png", "test")
        self.path.write_text("not an image", encoding="ascii")
        with self.assertRaises(ValueError):
            self.add()
        self.assertIsNone(self.world.get_meta(self.memory.META_KEY))

    def test_in_memory_db_and_small_crop(self):
        world = WorldDB(":memory:")
        try:
            memory = AppearanceMemory(world)
            memory.add("door", self.path, "detail", original=True,
                       bbox_normalized=[0, 0, 0.001, 0.001])
            result = memory.prompt_and_images([{"object_id": "door"}])
            self.assertEqual(result["images"][0].size, (1, 1))
            self.assertIsNone(memory.directory)
        finally:
            world.connection.close()

    def test_identity_does_not_claim_matching_state(self):
        self.add(original=True, state="closed")
        refs = self.memory.select("door", state="open")
        self.assertEqual(refs[0]["role"], "identity_only")
        self.assertEqual(refs[0]["state"], "closed")
        self.assertIn("No accepted appearance reference", self.memory.prompt_and_images(
            [{"object_id": "missing"}])["prompt"])

    def test_multiple_objects_keep_images_and_prompts_aligned(self):
        door = self.add(accepted=True, view="front")
        lamp = self.memory.add("lamp", self.path, "Blue lamp", accepted=True,
                               view="side", state="on", bbox_normalized=[0.5, 0, 1, 1])
        result = self.memory.prompt_and_images([
            {"object_id": "lamp", "view": "side", "state": "on"},
            {"object_id": "door", "view": "front"},
        ])
        self.assertEqual([r["id"] for r in result["refs"]], [lamp["id"], door["id"]])
        self.assertEqual([image.size for image in result["images"]], [(10, 10), (20, 10)])
        self.assertLess(result["prompt"].index("Blue lamp"), result["prompt"].index("A painted door"))

    def test_metadata_participates_in_world_transaction(self):
        original = self.add(original=True)
        with self.assertRaises(RuntimeError):
            with self.world.transaction():
                self.add(accepted=True)
                raise RuntimeError("rollback")
        self.assertEqual(self.world.get_meta(self.memory.META_KEY), [original])

    def test_quality_and_timestamp_break_equal_view_ties(self):
        self.add(accepted=True, view="front", quality=0.5, timestamp=100)
        older = self.add(accepted=True, view="front", quality=1, timestamp=1)
        newer = self.add(accepted=True, view="front", quality=1, timestamp=2)
        self.assertEqual([r["id"] for r in self.memory.select("door", "front")],
                         [newer["id"], older["id"]])


if __name__ == "__main__":
    unittest.main()
