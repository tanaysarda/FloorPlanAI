"""SYNTHETIC TEST DATA GENERATOR.

Creates small deterministic floor plans + label masks + an independent ground-truth file. Used only for
development and automated tests. Nothing produced here is, or may be presented as, Kaggle dataset data."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import SYNTHETIC_DIR

BG, WALL, DOOR, WINDOW = 0, 1, 2, 3
LABEL_IDS = {"living room": 10, "bedroom": 11, "kitchen": 12, "bathroom": 13, "hallway": 14}
NO_WINDOW_LABELS = {"bathroom", "hallway"}
T = 6          # wall thickness
M = 12         # background margin
DOOR_LEN, WIN_LEN, ENTRANCE_LEN = 24, 32, 28

# name, seed, rows, cols, labels in row-major order
SPECS: List[Tuple[str, int, int, int, List[str]]] = [
    ("synthetic_plan_01", 101, 1, 2, ["living room", "bathroom"]),
    ("synthetic_plan_02", 102, 2, 2, ["living room", "kitchen", "bedroom", "bathroom"]),
    ("synthetic_plan_03", 103, 2, 3, ["living room", "kitchen", "bedroom", "bathroom", "bedroom", "hallway"]),
    ("synthetic_plan_04", 104, 1, 3, ["bedroom", "hallway", "kitchen"]),
    ("synthetic_plan_05", 105, 2, 2, ["bedroom", "bedroom", "living room", "bathroom"]),
    ("synthetic_plan_06", 106, 2, 4, ["living room", "kitchen", "bedroom", "bathroom",
                                      "bedroom", "hallway", "bedroom", "bathroom"]),
    ("synthetic_plan_07", 107, 1, 5, ["bedroom", "bathroom", "living room", "kitchen", "bedroom"]),
    ("synthetic_plan_08", 108, 3, 2, ["living room", "kitchen", "hallway", "bedroom", "bathroom", "bedroom"]),
]


def _band(i: int, n: int, cuts: List[int], total: int) -> Tuple[int, int]:
    if i == 0:
        return 0, T
    if i == n:
        return total - T, total
    return cuts[i] - T // 2, cuts[i] - T // 2 + T


def _axis(total: int, n: int, rng: random.Random):
    cuts = [0] + [int(round(total * i / n + rng.uniform(-0.12, 0.12) * total / n)) for i in range(1, n)] + [total]
    bands = [_band(i, n, cuts, total) for i in range(n + 1)]
    return [(bands[i][1], bands[i + 1][0]) for i in range(n)], bands


def generate_plan(name: str, seed: int, rows: int, cols: int, labels: List[str]):
    """Return (image_bgr, mask_uint8, ground_truth_dict). Coordinates include the margin M."""
    rng = random.Random(seed)
    W, H = cols * 150, rows * 120
    xs, xb = _axis(W, cols, rng)
    ys, yb = _axis(H, rows, rng)
    mask = np.zeros((H + 2 * M, W + 2 * M), np.uint8)
    mask[M:M + H, M:M + W] = WALL
    cell = {}
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            (x0, x1), (y0, y1) = xs[c], ys[r]
            mask[M + y0:M + y1, M + x0:M + x1] = LABEL_IDS[labels[idx]]
            cell[(r, c)] = (idx, x0, x1, y0, y1)
    # adjacency (door candidates)
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1), "h"))
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c), "v"))
    rng.shuffle(edges)
    parent = {k: k for k in cell}
    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k
    chosen = []
    for a, b, o in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            chosen.append((a, b, o))
        elif rng.random() < 0.35:
            chosen.append((a, b, o))
    doors, windows = [], []
    for a, b, o in chosen:
        ia, ax0, ax1, ay0, ay1 = cell[a]
        ib, bx0, bx1, by0, by1 = cell[b]
        if o == "h":
            lo, hi = max(ay0, by0), min(ay1, by1)
            cy = (lo + hi) // 2
            mask[M + cy - DOOR_LEN // 2:M + cy + DOOR_LEN // 2, M + ax1:M + bx0] = DOOR
        else:
            lo, hi = max(ax0, bx0), min(ax1, bx1)
            cx = (lo + hi) // 2
            mask[M + ay1:M + by0, M + cx - DOOR_LEN // 2:M + cx + DOOR_LEN // 2] = DOOR
        assert hi - lo >= 40
        doors.append({"rooms": sorted([ia, ib]), "kind": "interior"})
    # entrance on the bottom wall of the bottom-left room
    ie, ex0, ex1, _, _ = cell[(rows - 1, 0)]
    cx = (ex0 + ex1) // 2
    mask[M + H - T:M + H, M + cx - ENTRANCE_LEN // 2:M + cx + ENTRANCE_LEN // 2] = DOOR
    doors.append({"rooms": [ie], "kind": "entrance"})
    # windows on one exterior wall of each eligible room
    for (r, c), (idx, x0, x1, y0, y1) in cell.items():
        if labels[idx] in NO_WINDOW_LABELS:
            continue
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if r == 0:
            mask[M:M + T, M + cx - WIN_LEN // 2:M + cx + WIN_LEN // 2] = WINDOW
        elif c == 0:
            mask[M + cy - WIN_LEN // 2:M + cy + WIN_LEN // 2, M:M + T] = WINDOW
        elif r == rows - 1:
            mask[M + H - T:M + H, M + cx - WIN_LEN // 2:M + cx + WIN_LEN // 2] = WINDOW
        elif c == cols - 1:
            mask[M + cy - WIN_LEN // 2:M + cy + WIN_LEN // 2, M + W - T:M + W] = WINDOW
        else:
            continue  # interior room: no exterior wall
        windows.append({"room": idx})
    rooms_gt = []
    for r in range(rows):
        for c in range(cols):
            idx, x0, x1, y0, y1 = cell[(r, c)]
            rooms_gt.append({"cell_index": idx, "label": labels[idx], "area_px": (x1 - x0) * (y1 - y0),
                             "bbox": [M + x0, M + y0, x1 - x0, y1 - y0]})
    areas = [r["area_px"] for r in rooms_gt]
    assert areas.count(max(areas)) == 1 and areas.count(min(areas)) == 1, f"area tie in {name}"
    gt = {"name": name, "rooms": rooms_gt, "doors": doors, "windows": windows}
    return render_image(mask, labels_by_bbox=rooms_gt), mask, gt


def render_image(mask: np.ndarray, labels_by_bbox) -> np.ndarray:
    """Draw a plain floor-plan-looking picture (white rooms, black walls) from the mask."""
    img = np.full(mask.shape + (3,), 255, np.uint8)
    img[mask == WALL] = (35, 35, 35)
    img[mask == DOOR] = (80, 140, 200)
    img[mask == WINDOW] = (235, 205, 140)
    for r in labels_by_bbox:
        x, y, w, h = r["bbox"]
        cv2.putText(img, r["label"], (x + 6, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1,
                    cv2.LINE_AA)
    return img


def generate_fixtures(out_dir: Path | str = SYNTHETIC_DIR) -> Dict:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    plans = []
    for name, seed, rows, cols, labels in SPECS:
        img, mask, gt = generate_plan(name, seed, rows, cols, labels)
        cv2.imwrite(str(out / "images" / f"{name}.png"), img)
        cv2.imwrite(str(out / "masks" / f"{name}.png"), mask)
        plans.append(gt)
    doc = {"synthetic": True, "note": "SYNTHETIC TEST DATA. Independent ground truth for tests/evaluation.",
           "plans": plans}
    (out / "ground_truth.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# SYNTHETIC TEST DATA\n\nGenerated by `processing/synthetic.py` for development and tests only.\n"
        "These files are NOT from the Kaggle floor plan dataset. Mask values: 0 background, 1 wall, 2 door, "
        "3 window, 10-14 room types (see configs/class_map_synthetic.json).\n", encoding="utf-8")
    return doc
