"""SQLite access layer: connection, initialisation, idempotent ingestion, JSON export/import."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import SCHEMA_PATH
from processing.models import FloorPlanData, Opening, Room, Wall


class DatabaseError(Exception):
    """Raised for database problems; messages are safe to show to users."""


def connect(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    try:
        if read_only:
            p = Path(path)
            if not p.is_file():
                raise DatabaseError("The database file does not exist yet.")
            conn = sqlite3.connect(f"file:{p.resolve().as_posix()}?mode=ro", uri=True)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except sqlite3.Error as exc:
        raise DatabaseError("Could not open the database.") from exc


def init_db(conn: sqlite3.Connection, dataset_kind: Optional[str] = None) -> None:
    """Create tables/views/indexes (idempotent)."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    set_meta(conn, "schema_version", "1")
    if dataset_kind:
        set_meta(conn, "dataset_kind", dataset_kind)


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def _validate_plan(plan: FloorPlanData) -> None:
    def unique(items, attr, what):
        idx = [getattr(i, attr) for i in items]
        if len(idx) != len(set(idx)):
            raise DatabaseError(f"duplicate {what} indices in plan {plan.name!r}")
    unique(plan.rooms, "index", "room"); unique(plan.doors, "index", "door")
    unique(plan.windows, "index", "window"); unique(plan.walls, "index", "wall")
    room_ids = {r.index for r in plan.rooms}
    for o in list(plan.doors) + list(plan.windows):
        if not set(o.room_indices) <= room_ids:
            raise DatabaseError(f"{o.kind} {o.index} references unknown room in plan {plan.name!r}")
    door_ids = {d.index for d in plan.doors}
    for a, b, d in plan.connections:
        if a not in room_ids or b not in room_ids or d not in door_ids or a == b:
            raise DatabaseError(f"invalid room connection in plan {plan.name!r}")


def upsert_floor_plan(conn: sqlite3.Connection, plan: FloorPlanData, plan_id: Optional[int] = None) -> int:
    """Insert or replace one plan (matched on dataset+name). Re-running never duplicates rows."""
    _validate_plan(plan)
    try:
        with conn:
            row = conn.execute("SELECT id FROM floor_plans WHERE dataset = ? AND name = ?",
                               (plan.dataset, plan.name)).fetchone()
            existing = row[0] if row else None
            if plan_id is None:
                plan_id = existing
            if plan_id is None:
                plan_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM floor_plans").fetchone()[0]
            else:
                clash = conn.execute("SELECT dataset, name FROM floor_plans WHERE id = ?", (plan_id,)).fetchone()
                if clash and tuple(clash) != (plan.dataset, plan.name):
                    raise DatabaseError(f"plan id {plan_id} already belongs to a different floor plan")
            if existing is not None:
                conn.execute("DELETE FROM floor_plans WHERE id = ?", (existing,))  # children cascade
            conn.execute(
                "INSERT INTO floor_plans(id, dataset, name, image_path, mask_path, width, height, is_synthetic,"
                " scale_m_per_px, wall_area_px, unknown_pixel_fraction, warnings_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (plan_id, plan.dataset, plan.name, plan.image_path, plan.mask_path, plan.width, plan.height,
                 int(plan.is_synthetic), plan.scale_m_per_px, plan.wall_area_px, plan.unknown_pixel_fraction,
                 json.dumps(plan.warnings)))
            room_db: Dict[int, int] = {}
            for r in plan.rooms:
                cur = conn.execute(
                    "INSERT INTO rooms(plan_id, room_index, label, class_id, area_px, bbox_x, bbox_y, bbox_w, bbox_h,"
                    " centroid_x, centroid_y) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (plan_id, r.index, r.label, r.class_id, r.area_px, *r.bbox, *r.centroid))
                room_db[r.index] = cur.lastrowid
            door_db: Dict[int, int] = {}
            for table, items, link, col in (("doors", plan.doors, "door_rooms", "door"),
                                            ("windows", plan.windows, "window_rooms", "window")):
                for o in items:
                    cur = conn.execute(
                        f"INSERT INTO {table}(plan_id, {col}_index, area_px, bbox_x, bbox_y, bbox_w, bbox_h,"
                        " centroid_x, centroid_y) VALUES (?,?,?,?,?,?,?,?,?)",
                        (plan_id, o.index, o.area_px, *o.bbox, *o.centroid))
                    if col == "door":
                        door_db[o.index] = cur.lastrowid
                    for ri in o.room_indices:
                        conn.execute(f"INSERT INTO {link}({col}_id, room_id) VALUES (?,?)",
                                     (cur.lastrowid, room_db[ri]))
            for w in plan.walls:
                conn.execute("INSERT INTO walls(plan_id, wall_index, area_px, bbox_x, bbox_y, bbox_w, bbox_h)"
                             " VALUES (?,?,?,?,?,?,?)", (plan_id, w.index, w.area_px, *w.bbox))
            for a, b, d in plan.connections:
                lo, hi = sorted((room_db[a], room_db[b]))
                conn.execute("INSERT OR IGNORE INTO room_connections(plan_id, room_a_id, room_b_id, door_id)"
                             " VALUES (?,?,?,?)", (plan_id, lo, hi, door_db[d]))
        return int(plan_id)
    except sqlite3.Error as exc:
        raise DatabaseError(f"could not store plan {plan.name!r}: {exc}") from exc


def ingest_plans(conn: sqlite3.Connection, plans: Iterable[FloorPlanData]) -> List[int]:
    """Ingest in name order so first-run plan numbers are deterministic."""
    return [upsert_floor_plan(conn, p) for p in sorted(plans, key=lambda p: (p.dataset, p.name))]


def list_plan_ids(conn: sqlite3.Connection) -> List[int]:
    return [r[0] for r in conn.execute("SELECT id FROM floor_plans ORDER BY id")]


def load_plan(conn: sqlite3.Connection, plan_id: int) -> Optional[FloorPlanData]:
    p = conn.execute("SELECT id, dataset, name, image_path, mask_path, width, height, is_synthetic,"
                     " scale_m_per_px, wall_area_px, unknown_pixel_fraction, warnings_json"
                     " FROM floor_plans WHERE id = ?", (plan_id,)).fetchone()
    if not p:
        return None
    rooms, by_id = [], {}
    for r in conn.execute("SELECT id, room_index, label, class_id, area_px, bbox_x, bbox_y, bbox_w, bbox_h,"
                          " centroid_x, centroid_y FROM rooms WHERE plan_id = ? ORDER BY room_index", (plan_id,)):
        rooms.append(Room(r[1], r[2], r[3], r[4], (r[5], r[6], r[7], r[8]), (r[9], r[10])))
        by_id[r[0]] = r[1]
    def openings(table, col, link, kind):
        out = []
        for o in conn.execute(f"SELECT id, {col}_index, area_px, bbox_x, bbox_y, bbox_w, bbox_h, centroid_x,"
                              f" centroid_y FROM {table} WHERE plan_id = ? ORDER BY {col}_index", (plan_id,)):
            ris = sorted(by_id[x[0]] for x in conn.execute(f"SELECT room_id FROM {link} WHERE {col}_id = ?", (o[0],)))
            out.append(Opening(o[1], kind, -1, o[2], (o[3], o[4], o[5], o[6]), (o[7], o[8]), ris))
        return out
    doors = openings("doors", "door", "door_rooms", "door")
    windows = openings("windows", "window", "window_rooms", "window")
    door_idx = {r[0]: r[1] for r in conn.execute("SELECT id, door_index FROM doors WHERE plan_id = ?", (plan_id,))}
    conns = sorted((by_id[a], by_id[b], door_idx[d]) for a, b, d in conn.execute(
        "SELECT room_a_id, room_b_id, door_id FROM room_connections WHERE plan_id = ?", (plan_id,)))
    walls = [Wall(w[0], w[1], (w[2], w[3], w[4], w[5])) for w in conn.execute(
        "SELECT wall_index, area_px, bbox_x, bbox_y, bbox_w, bbox_h FROM walls WHERE plan_id = ? ORDER BY wall_index",
        (plan_id,))]
    return FloorPlanData(name=p[2], dataset=p[1], width=p[5], height=p[6], image_path=p[3], mask_path=p[4],
                         is_synthetic=bool(p[7]), plan_id=p[0], scale_m_per_px=p[8], rooms=rooms, doors=doors,
                         windows=windows, walls=walls, wall_area_px=p[9], unknown_pixel_fraction=p[10],
                         connections=conns, warnings=json.loads(p[11]))


def export_json(conn: sqlite3.Connection, path: Path | str) -> int:
    plans = [load_plan(conn, i).to_dict() for i in list_plan_ids(conn)]
    doc = {"dataset_kind": get_meta(conn, "dataset_kind"), "plans": plans}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return len(plans)


def import_json(conn: sqlite3.Connection, path: Path | str) -> List[int]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        records = doc["plans"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DatabaseError("The JSON file is missing or malformed.") from exc
    ids = []
    for rec in records:
        try:
            plan = FloorPlanData.from_dict(rec)
        except ValueError as exc:
            raise DatabaseError(str(exc)) from exc
        ids.append(upsert_floor_plan(conn, plan, plan.plan_id))
    if doc.get("dataset_kind"):
        set_meta(conn, "dataset_kind", doc["dataset_kind"])
    return ids


def dataset_summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    one = lambda q: conn.execute(q).fetchone()[0]
    return {
        "dataset_kind": get_meta(conn, "dataset_kind"),
        "plans": one("SELECT COUNT(*) FROM floor_plans"), "rooms": one("SELECT COUNT(*) FROM rooms"),
        "doors": one("SELECT COUNT(*) FROM doors"), "windows": one("SELECT COUNT(*) FROM windows"),
        "walls": one("SELECT COUNT(*) FROM walls"),
        "room_labels": {r[0] if r[0] is not None else "(unlabelled)": r[1] for r in conn.execute(
            "SELECT label, COUNT(*) FROM rooms GROUP BY label ORDER BY COUNT(*) DESC, label")},
    }


def rows_to_dataframe(columns: List[str], rows: List[tuple]):
    import pandas as pd
    return pd.DataFrame(rows, columns=columns)
