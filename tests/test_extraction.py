import unittest

import numpy as np

from processing.extraction import extract_plan
from tests.helpers import CLASS_MAP, Workspace
from database import db


class ExtractionAgainstGroundTruth(unittest.TestCase):
    """Extraction output must match the generator's independent ground truth (synthetic data only)."""

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.plans = [db.load_plan(cls.ws.conn, i) for i in db.list_plan_ids(cls.ws.conn)]

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_all_plans_processed(self):
        self.assertEqual(len(self.plans), 8)
        self.assertEqual(self.ws.report.skipped, [])

    def test_rooms_match_ground_truth(self):
        for plan, gt in zip(self.plans, sorted(self.ws.gt["plans"], key=lambda p: p["name"])):
            self.assertEqual(len(plan.rooms), len(gt["rooms"]), plan.name)
            for room, g in zip(plan.rooms, gt["rooms"]):
                self.assertEqual(room.label, g["label"], plan.name)
                self.assertEqual(room.area_px, g["area_px"], plan.name)
                self.assertEqual(list(room.bbox), g["bbox"], plan.name)

    def test_doors_and_windows_match_ground_truth(self):
        for plan, gt in zip(self.plans, sorted(self.ws.gt["plans"], key=lambda p: p["name"])):
            want_doors = sorted(tuple(i + 1 for i in d["rooms"]) for d in gt["doors"])
            got_doors = sorted(tuple(d.room_indices) for d in plan.doors)
            self.assertEqual(got_doors, want_doors, plan.name)
            want_win = sorted(w["room"] + 1 for w in gt["windows"])
            got_win = sorted(r for w in plan.windows for r in w.room_indices)
            self.assertEqual(got_win, want_win, plan.name)
            self.assertTrue(all(len(w.room_indices) == 1 for w in plan.windows))

    def test_connections_derived_from_two_room_doors(self):
        for plan in self.plans:
            two = [d for d in plan.doors if len(d.room_indices) == 2]
            self.assertEqual(len(plan.connections), len(two))

    def test_no_real_world_units_claimed(self):
        for plan in self.plans:
            self.assertIsNone(plan.scale_m_per_px)
            self.assertTrue(plan.is_synthetic)


class ExtractionEdgeCases(unittest.TestCase):
    def test_empty_mask_yields_warning_not_crash(self):
        plan, maps = extract_plan("empty", np.zeros((30, 30), np.int32), CLASS_MAP, dataset="t")
        self.assertEqual((len(plan.rooms), len(plan.doors), len(plan.windows)), (0, 0, 0))
        self.assertIn("no rooms detected", plan.warnings)

    def test_unknown_fraction_warning(self):
        ids = np.full((10, 10), -1, np.int32)
        plan, _ = extract_plan("unk", ids, CLASS_MAP, dataset="t", unknown_fraction=1.0)
        self.assertTrue(any("not in the class map" in w for w in plan.warnings))

    def test_opening_with_no_adjacent_room_has_no_association(self):
        ids = np.zeros((60, 60), np.int32)
        ids[5:15, 5:15] = 10                 # room
        ids[40:44, 40:50] = 3                # far-away window
        plan, _ = extract_plan("far", ids, CLASS_MAP, dataset="t")
        self.assertEqual(plan.windows[0].room_indices, [])

    def test_class_map_without_rooms(self):
        from processing.segmentation import ClassMap, ClassSpec
        cm = ClassMap("w", "label_id", [ClassSpec(1, "wall", "wall")])
        plan, _ = extract_plan("x", np.ones((10, 10), np.int32), cm, dataset="t")
        self.assertTrue(any("no room classes" in w for w in plan.warnings))


if __name__ == "__main__":
    unittest.main()
