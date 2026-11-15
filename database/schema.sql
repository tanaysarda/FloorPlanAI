-- Floor plan database schema (SQLite). Safe to run repeatedly (IF NOT EXISTS).
-- All areas are PIXEL areas. scale_m_per_px is NULL unless a real scale is known.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS floor_plans (
    id                     INTEGER PRIMARY KEY,           -- the "plan number" users refer to
    dataset                TEXT NOT NULL,
    name                   TEXT NOT NULL,                 -- source file stem
    image_path             TEXT,
    mask_path              TEXT,
    width                  INTEGER NOT NULL CHECK (width > 0),
    height                 INTEGER NOT NULL CHECK (height > 0),
    is_synthetic           INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
    scale_m_per_px         REAL CHECK (scale_m_per_px IS NULL OR scale_m_per_px > 0),
    wall_area_px           INTEGER NOT NULL DEFAULT 0 CHECK (wall_area_px >= 0),
    unknown_pixel_fraction REAL NOT NULL DEFAULT 0 CHECK (unknown_pixel_fraction BETWEEN 0 AND 1),
    warnings_json          TEXT NOT NULL DEFAULT '[]',
    UNIQUE (dataset, name)
);

CREATE TABLE IF NOT EXISTS rooms (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id    INTEGER NOT NULL REFERENCES floor_plans(id) ON DELETE CASCADE,
    room_index INTEGER NOT NULL CHECK (room_index >= 1),
    label      TEXT,
    class_id   INTEGER NOT NULL,
    area_px    INTEGER NOT NULL CHECK (area_px >= 0),
    bbox_x INTEGER NOT NULL, bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL CHECK (bbox_w >= 0), bbox_h INTEGER NOT NULL CHECK (bbox_h >= 0),
    centroid_x REAL NOT NULL, centroid_y REAL NOT NULL,
    UNIQUE (plan_id, room_index)
);

CREATE TABLE IF NOT EXISTS doors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id    INTEGER NOT NULL REFERENCES floor_plans(id) ON DELETE CASCADE,
    door_index INTEGER NOT NULL CHECK (door_index >= 1),
    area_px    INTEGER NOT NULL CHECK (area_px >= 0),
    bbox_x INTEGER NOT NULL, bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL CHECK (bbox_w >= 0), bbox_h INTEGER NOT NULL CHECK (bbox_h >= 0),
    centroid_x REAL NOT NULL, centroid_y REAL NOT NULL,
    UNIQUE (plan_id, door_index)
);

CREATE TABLE IF NOT EXISTS windows (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id      INTEGER NOT NULL REFERENCES floor_plans(id) ON DELETE CASCADE,
    window_index INTEGER NOT NULL CHECK (window_index >= 1),
    area_px      INTEGER NOT NULL CHECK (area_px >= 0),
    bbox_x INTEGER NOT NULL, bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL CHECK (bbox_w >= 0), bbox_h INTEGER NOT NULL CHECK (bbox_h >= 0),
    centroid_x REAL NOT NULL, centroid_y REAL NOT NULL,
    UNIQUE (plan_id, window_index)
);

-- One row per connected wall region (not per straight wall segment).
CREATE TABLE IF NOT EXISTS walls (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id    INTEGER NOT NULL REFERENCES floor_plans(id) ON DELETE CASCADE,
    wall_index INTEGER NOT NULL CHECK (wall_index >= 1),
    area_px    INTEGER NOT NULL CHECK (area_px >= 0),
    bbox_x INTEGER NOT NULL, bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL CHECK (bbox_w >= 0), bbox_h INTEGER NOT NULL CHECK (bbox_h >= 0),
    UNIQUE (plan_id, wall_index)
);

-- Many-to-many: an opening is associated with a room when its mask touches the room mask.
CREATE TABLE IF NOT EXISTS door_rooms (
    door_id INTEGER NOT NULL REFERENCES doors(id) ON DELETE CASCADE,
    room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    PRIMARY KEY (door_id, room_id)
);
CREATE TABLE IF NOT EXISTS window_rooms (
    window_id INTEGER NOT NULL REFERENCES windows(id) ON DELETE CASCADE,
    room_id   INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    PRIMARY KEY (window_id, room_id)
);

-- Pairs of rooms joined by a door (derived from door_rooms; useful for graph questions).
CREATE TABLE IF NOT EXISTS room_connections (
    plan_id   INTEGER NOT NULL REFERENCES floor_plans(id) ON DELETE CASCADE,
    room_a_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    room_b_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    door_id   INTEGER NOT NULL REFERENCES doors(id) ON DELETE CASCADE,
    CHECK (room_a_id < room_b_id),
    PRIMARY KEY (door_id, room_a_id, room_b_id)
);

CREATE INDEX IF NOT EXISTS idx_rooms_plan    ON rooms(plan_id);
CREATE INDEX IF NOT EXISTS idx_rooms_label   ON rooms(label);
CREATE INDEX IF NOT EXISTS idx_doors_plan    ON doors(plan_id);
CREATE INDEX IF NOT EXISTS idx_windows_plan  ON windows(plan_id);
CREATE INDEX IF NOT EXISTS idx_walls_plan    ON walls(plan_id);
CREATE INDEX IF NOT EXISTS idx_door_rooms_room   ON door_rooms(room_id);
CREATE INDEX IF NOT EXISTS idx_window_rooms_room ON window_rooms(room_id);
CREATE INDEX IF NOT EXISTS idx_conn_plan     ON room_connections(plan_id);

CREATE VIEW IF NOT EXISTS plan_summary AS
SELECT p.id AS plan_id,
       p.name AS name,
       (SELECT COUNT(*) FROM rooms   r WHERE r.plan_id = p.id) AS room_count,
       (SELECT COUNT(*) FROM doors   d WHERE d.plan_id = p.id) AS door_count,
       (SELECT COUNT(*) FROM windows w WHERE w.plan_id = p.id) AS window_count,
       (SELECT COUNT(*) FROM walls   l WHERE l.plan_id = p.id) AS wall_count,
       (SELECT COALESCE(SUM(r.area_px), 0) FROM rooms r WHERE r.plan_id = p.id) AS total_room_area_px
FROM floor_plans p;

CREATE VIEW IF NOT EXISTS room_details AS
SELECT r.id AS room_id, r.plan_id, r.room_index, r.label,
       COALESCE(r.label, 'room') || ' #' || r.room_index AS room_name,
       r.area_px, r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h, r.centroid_x, r.centroid_y,
       (SELECT COUNT(*) FROM window_rooms wr WHERE wr.room_id = r.id) AS window_count,
       (SELECT COUNT(*) FROM door_rooms   dr WHERE dr.room_id = r.id) AS door_count
FROM rooms r;
