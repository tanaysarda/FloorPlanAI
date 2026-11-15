import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from database import db
from processing.models import FloorPlanData, Opening, Room
from processing.pipeline import ingest_dataset
from tests.helpers import CLASS_MAP, Workspace, table_counts


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.close()

    def test_reingestion_creates_no_duplicates_and_keeps_ids(self):
        before = table_counts(self.ws.conn)
        ids_before = db.list_plan_ids(self.ws.conn)
        for _ in range(2):
            ingest_dataset(self.ws.adapter, CLASS_MAP, self.ws.conn)
        self.assertEqual(table_counts(self.ws.conn), before)
        self.assertEqual(db.list_plan_ids(self.ws.conn), ids_before)
        self.assertEqual(ids_before, list(range(1, 9)))

    def test_init_is_repeatable(self):
        db.init_db(self.ws.conn, "synthetic")
        db.init_db(self.ws.conn, "synthetic")
        self.assertEqual(db.get_meta(self.ws.conn, "dataset_kind"), "synthetic")

    def test_foreign_keys_and_checks_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.ws.conn.execute("INSERT INTO rooms(plan_id, room_index, class_id, area_px, bbox_x, bbox_y,"
                                 " bbox_w, bbox_h, centroid_x, centroid_y) VALUES (999,1,1,5,0,0,1,1,0,0)")
        with self.assertRaises(sqlite3.IntegrityError):
            self.ws.conn.execute("INSERT INTO rooms(plan_id, room_index, class_id, area_px, bbox_x, bbox_y,"
                                 " bbox_w, bbox_h, centroid_x, centroid_y) VALUES (1,99,1,-5,0,0,1,1,0,0)")
        self.ws.conn.rollback()

    def test_views_agree_with_tables(self):
        for pid, rc, dc, wc in self.ws.conn.execute("SELECT plan_id, room_count, door_count, window_count FROM plan_summary"):
            self.assertEqual(rc, self.ws.conn.execute("SELECT COUNT(*) FROM rooms WHERE plan_id=?", (pid,)).fetchone()[0])
            self.assertEqual(dc, self.ws.conn.execute("SELECT COUNT(*) FROM doors WHERE plan_id=?", (pid,)).fetchone()[0])
            self.assertEqual(wc, self.ws.conn.execute("SELECT COUNT(*) FROM windows WHERE plan_id=?", (pid,)).fetchone()[0])

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = Path(d) / "a.json", Path(d) / "b.json"
            db.export_json(self.ws.conn, p1)
            conn2 = db.connect(Path(d) / "copy.db")
            db.init_db(conn2)
            db.import_json(conn2, p1)
            db.export_json(conn2, p2)
            self.assertEqual(json.loads(p1.read_text()), json.loads(p2.read_text()))
            self.assertEqual(table_counts(conn2), table_counts(self.ws.conn))
            conn2.close()

    def test_malformed_json_import(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.json"
            bad.write_text("{not json")
            conn = db.connect(Path(d) / "x.db"); db.init_db(conn)
            with self.assertRaises(db.DatabaseError):
                db.import_json(conn, bad)
            bad.write_text(json.dumps({"plans": [{"name": "x"}]}))
            with self.assertRaises(db.DatabaseError):
                db.import_json(conn, bad)
            with self.assertRaises(db.DatabaseError):
                db.import_json(conn, Path(d) / "missing.json")
            conn.close()

    def test_invalid_plan_records_rejected(self):
        bad = FloorPlanData("bad", "t", 10, 10, rooms=[Room(1, "a", 1, 5, (0, 0, 1, 1), (0, 0))],
                            doors=[Opening(1, "door", 2, 3, (0, 0, 1, 1), (0, 0), [5])])
        with self.assertRaises(db.DatabaseError):
            db.upsert_floor_plan(self.ws.conn, bad)
        self.assertEqual(self.ws.conn.execute("SELECT COUNT(*) FROM floor_plans WHERE name='bad'").fetchone()[0], 0)

    def test_read_only_connection_cannot_write(self):
        ro = db.connect(self.ws.db_path, read_only=True)
        with self.assertRaises(sqlite3.OperationalError):
            ro.execute("DELETE FROM rooms")
        ro.close()

    def test_missing_db_read_only(self):
        with self.assertRaises(db.DatabaseError):
            db.connect(self.ws.root / "nope.db", read_only=True)

    def test_dataset_summary_and_dataframe(self):
        s = db.dataset_summary(self.ws.conn)
        self.assertEqual((s["plans"], s["rooms"]), (8, 38))
        df = db.rows_to_dataframe(["a", "b"], [(1, 2)])
        self.assertEqual(list(df.columns), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
