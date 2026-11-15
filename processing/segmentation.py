"""Class maps and mask processing (OpenCV + NumPy). No dataset-specific assumptions live here."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

VALID_KINDS = {"background", "wall", "door", "window", "room", "ignore"}
UNKNOWN = -1


class MaskFormatError(Exception):
    """The mask array does not match what the class map expects."""


@dataclass(frozen=True)
class ClassSpec:
    id: int
    name: str
    kind: str
    color: Optional[Tuple[int, int, int]] = None  # RGB, for colour-coded masks


@dataclass(frozen=True)
class Component:
    label: int          # id inside the cv2 label image
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]


class ClassMap:
    """Maps raw mask values (label ids or RGB colours) to semantic kinds (wall/door/window/room)."""

    def __init__(self, name: str, mask_mode: str, classes: Sequence[ClassSpec], color_tolerance: int = 0):
        if mask_mode not in ("label_id", "rgb"):
            raise ValueError("mask_mode must be 'label_id' or 'rgb'")
        ids = [c.id for c in classes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate class ids in class map")
        for c in classes:
            if c.kind not in VALID_KINDS:
                raise ValueError(f"class {c.name!r} has invalid kind {c.kind!r}; use one of {sorted(VALID_KINDS)}")
            if mask_mode == "rgb" and c.color is None:
                raise ValueError(f"class {c.name!r} needs a colour in rgb mode")
        self.name, self.mask_mode, self.color_tolerance = name, mask_mode, int(color_tolerance)
        self.classes = list(classes)
        self._by_id: Dict[int, ClassSpec] = {c.id: c for c in classes}

    @classmethod
    def from_dict(cls, d: dict) -> "ClassMap":
        try:
            specs = [ClassSpec(int(c["id"]), str(c["name"]), str(c["kind"]),
                               tuple(int(v) for v in c["color"]) if c.get("color") else None)
                     for c in d["classes"]]
            return cls(d.get("name", "unnamed"), d["mask_mode"], specs, d.get("color_tolerance", 0))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid class map: {exc!r}") from exc

    @classmethod
    def from_json(cls, path: Path | str) -> "ClassMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def spec(self, class_id: int) -> Optional[ClassSpec]:
        return self._by_id.get(class_id)

    def ids_for_kind(self, kind: str) -> List[int]:
        return [c.id for c in self.classes if c.kind == kind]

    def to_class_ids(self, raw: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert a raw mask into an int32 array of class ids (-1 = unknown) plus the unknown-pixel fraction."""
        if raw is None or raw.size == 0:
            raise MaskFormatError("mask is empty")
        if self.mask_mode == "label_id":
            if raw.ndim == 3:
                if not all(np.array_equal(raw[..., 0], raw[..., k]) for k in range(1, raw.shape[2])):
                    raise MaskFormatError("multi-channel mask is not a grey label image; configure mask_mode 'rgb'")
                raw = raw[..., 0]
            ids = np.full(raw.shape, UNKNOWN, dtype=np.int32)
            for cid in self._by_id:
                ids[raw == cid] = cid
        else:
            if raw.ndim != 3 or raw.shape[2] < 3:
                raise MaskFormatError("rgb mode needs a 3-channel mask")
            rgb = raw[..., :3][..., ::-1].astype(np.int16)  # cv2 loads BGR
            ids = np.full(raw.shape[:2], UNKNOWN, dtype=np.int32)
            for c in self.classes:
                diff = np.abs(rgb - np.array(c.color, dtype=np.int16)).max(axis=2)
                ids[diff <= self.color_tolerance] = c.id
        return ids, float((ids == UNKNOWN).mean())


def binary_for(class_ids: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    if not len(ids):
        return np.zeros(class_ids.shape, dtype=bool)
    return np.isin(class_ids, list(ids))


def connected_components(binary: np.ndarray, min_area: int = 1) -> Tuple[List[Component], np.ndarray]:
    """4-connected components with area >= min_area. Returns components and a label image (filtered ones = 0)."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=4)
    comps: List[Component] = []
    out = np.zeros(labels.shape, dtype=np.int32)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        comps.append(Component(i, area, (x, y, w, h), (float(cents[i][0]), float(cents[i][1]))))
        out[labels == i] = i
    return comps, out
