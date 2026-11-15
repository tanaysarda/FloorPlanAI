import io
import json
import py_compile
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import cli
import config
from database import db
from evaluation.evaluator import check, run_evaluation
from processing.loader import FolderPairAdapter
from processing.pipeline import ingest_dataset
from assistant.qa_engine import QAEngine, QAResult
from tests.helpers import CLASS_MAP, Workspace


class PipelineRobustness(unittest.TestCase):
    def test_bad_inputs_are_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "images").mkdir(); (d / "masks").mkdir()
            good = np.zeros((60, 60), np.uint8); good[5:30, 5:30] = 10
            def img(name): cv2.imwrite(str(d / "images" / f"{name}.png"), np.full((60, 60, 3), 255, np.uint8))
            img("a_ok"); cv2.imwrite(str(d / "masks" / "a_ok.png"), good)
            img("b_nomask")
            img("c_corrupt_mask"); (d / "masks" / "c_corrupt_mask.png").write_bytes(b"not a png at all")
            (d / "images" / "d_corrupt_image.png").write_bytes(b"garbage"); cv2.imwrite(str(d / "masks" / "d_corrupt_image.png"), good)
            img("e_unknown_labels"); odd = good.copy(); odd[40:50, 40:50] = 99; cv2.imwrite(str(d / "masks" / "e_unknown_labels.png"), odd)
            img("f_colour_mask"); col = np.dstack([good, good[::-1], good]); cv2.imwrite(str(d / "masks" / "f_colour_mask.png"), col)
            conn = db.connect(d / "t.db"); db.init_db(conn)
            rep = ingest_dataset(FolderPairAdapter(d / "images", d / "masks", "t"), CLASS_MAP, conn, overlays_dir=d / "ov")
            skipped = dict(rep.skipped)
            self.assertEqual(sorted(rep.processed), ["a_ok", "d_corrupt_image", "e_unknown_labels"])
            self.assertIn("no segmentation mask", skipped["b_nomask"])
            self.assertIn("corrupt", skipped["c_corrupt_mask"])
            self.assertIn("multi-channel", skipped["f_colour_mask"])
            plan = db.load_plan(conn, 2)
            self.assertTrue(any("corrupt" in w for w in plan.warnings))
            self.assertTrue(any("not in the class map" in w for w in db.load_plan(conn, 3).warnings))
            self.assertEqual(len(list((d / "ov").glob("*_overlay.png"))), 3)
            conn.close()

    def test_empty_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            conn = db.connect(Path(d) / "t.db"); db.init_db(conn)
            rep = ingest_dataset(FolderPairAdapter(Path(d) / "none", Path(d) / "none2", "t"), CLASS_MAP, conn)
            self.assertEqual((rep.processed, rep.skipped), ([], []))
            self.assertEqual(db.list_plan_ids(conn), [])
            conn.close()


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_three_or_more_overlays_are_real_images_with_rooms_drawn(self):
        files = sorted((self.ws.root / "overlays").glob("*_overlay.png"))
        self.assertGreaterEqual(len(files), 3)
        for f in files[:3]:
            ov = cv2.imread(str(f))
            self.assertIsNotNone(ov)
            orig = cv2.imread(str(self.ws.root / "data" / "images" / f.name.replace("_overlay", "")))
            self.assertEqual(ov.shape, orig.shape)
            self.assertGreater(int(np.abs(ov.astype(int) - orig.astype(int)).sum()), 0)

    def test_full_evaluation_on_synthetic_set(self):
        q = config.ROOT / "evaluation" / "test_questions_synthetic.json"
        report = run_evaluation(QAEngine(self.ws.db_path), q)
        s = report["summary"]
        self.assertGreaterEqual(s["total"], 20)
        self.assertEqual(s["correct"] + s["incorrect"], s["total"])
        self.assertEqual(s["accuracy_pct"], 100.0, report["failures"])
        self.assertEqual(run_evaluation(QAEngine(self.ws.db_path), q)["summary"], s)   # reproducible

    def test_evaluator_detects_wrong_expectations(self):
        res = QAResult("q", "answered", "x", "SELECT 1", (), ["value"], [(5,)])
        self.assertTrue(check({"type": "scalar", "column": "value", "value": 5}, res)[0])
        self.assertFalse(check({"type": "scalar", "column": "value", "value": 6}, res)[0])
        self.assertFalse(check({"type": "set", "column": "value", "values": [5, 6]}, res)[0])
        self.assertFalse(check({"type": "empty"}, res)[0])
        self.assertFalse(check({"type": "row", "values": {"value": 9}}, res)[0])

    def test_cli_ask_json_and_text(self):
        root = str(self.ws.root / "out")
        cfg = config.paths_for("synthetic", Path(root))
        cfg.output_dir.mkdir(parents=True)
        cfg.db_path.write_bytes(self.ws.db_path.read_bytes())
        buf = io.StringIO()
        code = cli.main(["ask", "How many rooms does plan 3 have?", "--no-llm", "--output-root", root, "--json"], buf)
        out = json.loads(buf.getvalue())
        self.assertEqual((code, out["status"], out["rows"]), (0, "answered", [[3, 6]]))
        buf = io.StringIO()
        cli.main(["ask", "How many rooms does plan 3 have?", "--no-llm", "--output-root", root], buf)
        self.assertIn("SYNTHETIC TEST DATA", buf.getvalue())
        self.assertIn("SQL:", buf.getvalue())

    def test_cli_missing_db_is_graceful(self):
        buf = io.StringIO()
        code = cli.main(["ask", "How many rooms does plan 3 have?", "--no-llm", "--output-root",
                         str(self.ws.root / "empty")], buf)
        self.assertEqual(code, 1)
        self.assertIn("not available", buf.getvalue())

    def test_streamlit_app_syntax_only(self):
        """Syntax check ONLY. Streamlit is not installed here, so the app has never been launched."""
        py_compile.compile(str(config.ROOT / "app.py"), doraise=True)


if __name__ == "__main__":
    unittest.main()
