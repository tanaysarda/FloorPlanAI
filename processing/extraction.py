"""Turn a class-id mask into structured rooms/doors/windows/walls (all measurements in pixels)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from processing.models import FloorPlanData, Opening, Room, Wall
from processing.segmentation import ClassMap, Component, binary_for, connected_components


@dataclass(frozen=True)
class ExtractionParams:
    min_room_area_px: int = 50
    min_opening_area_px: int = 4
    min_wall_area_px: int = 4
    assoc_radius_px: int = 5        # how far an opening is dilated to find the rooms it touches
    assoc_min_overlap_px: int = 3   # min touching pixels to associate an opening with a room


@dataclass
class InstanceMaps:
    """Per-pixel instance ids (0 = none); ids equal the 1-based record index."""
    rooms: np.ndarray
    doors: np.ndarray
    windows: np.ndarray
    walls: np.ndarray


def _reading_order(items):
    return sorted(items, key=lambda t: (t[1].bbox[1], t[1].bbox[0], t[0]))


def _associate(opening_mask: np.ndarray, bbox, room_inst: np.ndarray, params: ExtractionParams) -> List[int]:
    h, w = room_inst.shape
    r = params.assoc_radius_px
    x, y, bw, bh = bbox
    x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(w, x + bw + r), min(h, y + bh + r)
    crop = opening_mask[y0:y1, x0:x1].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near = cv2.dilate(crop, k) > 0
    ids, counts = np.unique(room_inst[y0:y1, x0:x1][near], return_counts=True)
    return sorted(int(i) for i, c in zip(ids, counts) if i > 0 and c >= params.assoc_min_overlap_px)


def extract_plan(name: str, class_ids: np.ndarray, class_map: ClassMap, *, dataset: str,
                 image_path: Optional[str] = None, mask_path: Optional[str] = None, is_synthetic: bool = False,
                 unknown_fraction: float = 0.0, params: ExtractionParams = ExtractionParams(),
                 warnings: Optional[List[str]] = None) -> Tuple[FloorPlanData, InstanceMaps]:
    h, w = class_ids.shape
    warns = list(warnings or [])
    if unknown_fraction > 0:
        warns.append(f"{unknown_fraction:.2%} of mask pixels have values not in the class map (ignored)")

    # --- rooms: components per room class so adjacent rooms of different types never merge
    found = []
    room_classes = class_map.ids_for_kind("room")
    if not room_classes:
        warns.append("class map defines no room classes; no rooms can be extracted")
    for cid in room_classes:
        comps, labs = connected_components(class_ids == cid, params.min_room_area_px)
        found += [((cid, labs), c) for c in comps]
    ordered = sorted(found, key=lambda t: (t[1].bbox[1], t[1].bbox[0], t[0][0]))
    room_inst = np.zeros((h, w), dtype=np.int32)
    rooms: List[Room] = []
    for idx, ((cid, labs), c) in enumerate(ordered, start=1):
        room_inst[labs == c.label] = idx
        rooms.append(Room(idx, class_map.spec(cid).name, cid, c.area, c.bbox, c.centroid))
    if not rooms:
        warns.append("no rooms detected")

    # --- walls
    wall_bin = binary_for(class_ids, class_map.ids_for_kind("wall"))
    wcomps, wlabs = connected_components(wall_bin, params.min_wall_area_px)
    wcomps = [c for _, c in _reading_order(list(enumerate(wcomps)))]
    wall_inst = np.zeros((h, w), dtype=np.int32)
    walls = []
    for idx, c in enumerate(wcomps, start=1):
        wall_inst[wlabs == c.label] = idx
        walls.append(Wall(idx, c.area, c.bbox))

    # --- doors / windows
    def openings(kind: str):
        binary = binary_for(class_ids, class_map.ids_for_kind(kind))
        comps, labs = connected_components(binary, params.min_opening_area_px)
        comps = [c for _, c in _reading_order(list(enumerate(comps)))]
        inst = np.zeros((h, w), dtype=np.int32)
        out = []
        for idx, c in enumerate(comps, start=1):
            m = labs == c.label
            inst[m] = idx
            out.append(Opening(idx, kind, -1, c.area, c.bbox, c.centroid, _associate(m, c.bbox, room_inst, params)))
        return out, inst
    doors, door_inst = openings("door")
    windows, window_inst = openings("window")
    for o in doors + windows:
        spec_ids = class_map.ids_for_kind(o.kind)
        o.class_id = spec_ids[0] if spec_ids else -1

    connections = sorted({(a, b, d.index) for d in doors for i, a in enumerate(d.room_indices)
                          for b in d.room_indices[i + 1:]})
    plan = FloorPlanData(name=name, dataset=dataset, width=w, height=h, image_path=image_path, mask_path=mask_path,
                         is_synthetic=is_synthetic, rooms=rooms, doors=doors, windows=windows, walls=walls,
                         wall_area_px=int(wall_bin.sum()), unknown_pixel_fraction=unknown_fraction,
                         connections=connections, warnings=warns)
    return plan, InstanceMaps(room_inst, door_inst, window_inst, wall_inst)
