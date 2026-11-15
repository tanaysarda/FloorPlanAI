"""Dataclasses describing the structured information extracted from one floor plan."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x, y, width, height (pixels)


@dataclass
class Room:
    index: int                      # 1-based, reading order (top-to-bottom, left-to-right)
    label: Optional[str]            # class name from the class map
    class_id: int
    area_px: int                    # PIXEL area. Never square metres.
    bbox: BBox
    centroid: Tuple[float, float]


@dataclass
class Opening:
    """A door or a window instance."""
    index: int
    kind: str                       # "door" | "window"
    class_id: int
    area_px: int
    bbox: BBox
    centroid: Tuple[float, float]
    room_indices: List[int] = field(default_factory=list)  # rooms it touches (may be empty)


@dataclass
class Wall:
    index: int
    area_px: int
    bbox: BBox


@dataclass
class FloorPlanData:
    name: str
    dataset: str
    width: int
    height: int
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    is_synthetic: bool = False
    plan_id: Optional[int] = None
    scale_m_per_px: Optional[float] = None   # None = unknown; no real-world units are ever claimed
    rooms: List[Room] = field(default_factory=list)
    doors: List[Opening] = field(default_factory=list)
    windows: List[Opening] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)
    wall_area_px: int = 0
    unknown_pixel_fraction: float = 0.0
    connections: List[Tuple[int, int, int]] = field(default_factory=list)  # (room_a, room_b, door_index)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FloorPlanData":
        try:
            def box(b): return tuple(int(v) for v in b)
            def pt(c): return (float(c[0]), float(c[1]))
            rooms = [Room(int(r["index"]), r.get("label"), int(r["class_id"]), int(r["area_px"]),
                          box(r["bbox"]), pt(r["centroid"])) for r in d.get("rooms", [])]
            def openings(items, kind):
                return [Opening(int(o["index"]), kind, int(o["class_id"]), int(o["area_px"]), box(o["bbox"]),
                                pt(o["centroid"]), [int(i) for i in o.get("room_indices", [])]) for o in items]
            return cls(
                name=str(d["name"]), dataset=str(d["dataset"]), width=int(d["width"]), height=int(d["height"]),
                image_path=d.get("image_path"), mask_path=d.get("mask_path"),
                is_synthetic=bool(d.get("is_synthetic", False)), plan_id=d.get("plan_id"),
                scale_m_per_px=d.get("scale_m_per_px"), rooms=rooms,
                doors=openings(d.get("doors", []), "door"), windows=openings(d.get("windows", []), "window"),
                walls=[Wall(int(w["index"]), int(w["area_px"]), box(w["bbox"])) for w in d.get("walls", [])],
                wall_area_px=int(d.get("wall_area_px", 0)),
                unknown_pixel_fraction=float(d.get("unknown_pixel_fraction", 0.0)),
                connections=[(int(a), int(b), int(c)) for a, b, c in d.get("connections", [])],
                warnings=[str(w) for w in d.get("warnings", [])],
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"malformed floor plan record: {exc!r}") from exc
