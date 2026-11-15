"""Shared test helpers. All data used here is SYNTHETIC TEST DATA."""
from __future__ import annotations

import tempfile
from pathlib import Path

import config
from database import db
from processing.loader import FolderPairAdapter
from processing.pipeline import ingest_dataset
from processing.segmentation import ClassMap
from processing.synthetic import generate_fixtures

CLASS_MAP = ClassMap.from_json(config.CONFIG_DIR / "class_map_synthetic.json")


class Workspace:
    """Temporary synthetic dataset + ingested database."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.gt = generate_fixtures(self.root / "data")
        self.db_path = self.root / "test.db"
        self.conn = db.connect(self.db_path)
        db.init_db(self.conn, "synthetic")
        self.adapter = FolderPairAdapter(self.root / "data" / "images", self.root / "data" / "masks",
                                         "synthetic", is_synthetic=True)
        self.report = ingest_dataset(self.adapter, CLASS_MAP, self.conn, overlays_dir=self.root / "overlays")

    def close(self):
        self.conn.close()
        self._tmp.cleanup()


def table_counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("floor_plans", "rooms", "doors", "windows", "walls", "door_rooms", "window_rooms",
                      "room_connections")}
