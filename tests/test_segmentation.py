import unittest

import numpy as np

from processing.segmentation import ClassMap, ClassSpec, MaskFormatError, connected_components


class ClassMapTests(unittest.TestCase):
    def test_label_mode_and_unknown(self):
        cm = ClassMap("t", "label_id", [ClassSpec(0, "bg", "background"), ClassSpec(1, "wall", "wall")])
        raw = np.array([[0, 1], [7, 1]], np.uint8)
        ids, unknown = cm.to_class_ids(raw)
        self.assertEqual(ids.tolist(), [[0, 1], [-1, 1]])
        self.assertAlmostEqual(unknown, 0.25)

    def test_rgb_mode_with_tolerance(self):
        specs = [ClassSpec(1, "wall", "wall", (255, 255, 255)), ClassSpec(4, "room", "room", (128, 128, 128))]
        raw_rgb = np.array([[[255, 255, 255], [130, 127, 128], [0, 0, 0]]], np.uint8)
        raw_bgr = raw_rgb[..., ::-1].copy()
        strict = ClassMap("t", "rgb", specs, 0).to_class_ids(raw_bgr)[0]
        self.assertEqual(strict.tolist(), [[1, -1, -1]])
        tol = ClassMap("t", "rgb", specs, 3).to_class_ids(raw_bgr)[0]
        self.assertEqual(tol.tolist(), [[1, 4, -1]])

    def test_bad_inputs(self):
        cm = ClassMap("t", "label_id", [ClassSpec(0, "bg", "background")])
        with self.assertRaises(MaskFormatError):
            cm.to_class_ids(np.zeros((0, 0), np.uint8))
        colour = np.zeros((2, 2, 3), np.uint8); colour[0, 0] = (1, 2, 3)
        with self.assertRaises(MaskFormatError):
            cm.to_class_ids(colour)
        with self.assertRaises(MaskFormatError):
            ClassMap("t", "rgb", [ClassSpec(1, "w", "wall", (1, 1, 1))]).to_class_ids(np.zeros((2, 2), np.uint8))

    def test_invalid_class_maps(self):
        with self.assertRaises(ValueError):
            ClassMap("t", "label_id", [ClassSpec(1, "x", "banana")])
        with self.assertRaises(ValueError):
            ClassMap("t", "label_id", [ClassSpec(1, "a", "wall"), ClassSpec(1, "b", "door")])
        with self.assertRaises(ValueError):
            ClassMap("t", "rgb", [ClassSpec(1, "a", "wall")])
        with self.assertRaises(ValueError):
            ClassMap.from_dict({"mask_mode": "label_id"})


class ComponentTests(unittest.TestCase):
    def test_components_and_min_area(self):
        b = np.zeros((20, 20), bool)
        b[2:6, 2:6] = True      # 16 px
        b[10:12, 10:12] = True  # 4 px
        comps, labs = connected_components(b, min_area=5)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0].area, 16)
        self.assertEqual(comps[0].bbox, (2, 2, 4, 4))
        self.assertEqual(int((labs > 0).sum()), 16)

    def test_empty_binary(self):
        comps, labs = connected_components(np.zeros((5, 5), bool))
        self.assertEqual(comps, [])
        self.assertEqual(int(labs.sum()), 0)

    def test_diagonal_pixels_are_separate_under_4_connectivity(self):
        b = np.zeros((4, 4), bool); b[0, 0] = b[1, 1] = True
        self.assertEqual(len(connected_components(b)[0]), 2)


if __name__ == "__main__":
    unittest.main()
