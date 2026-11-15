"""Overlay rendering (OpenCV) and side-by-side comparison figures (Matplotlib)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from processing.extraction import InstanceMaps
from processing.models import FloorPlanData

PALETTE_BGR = [(75, 180, 60), (200, 130, 0), (60, 60, 230), (0, 200, 230), (200, 60, 170), (190, 190, 50),
               (30, 130, 245), (130, 130, 255), (150, 100, 40), (80, 220, 160)]
DOOR_BGR, WINDOW_BGR = (0, 140, 255), (255, 160, 0)


def _text(img, txt, org, scale=0.45):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def render_overlay(image_bgr: Optional[np.ndarray], plan: FloorPlanData, maps: InstanceMaps,
                   alpha: float = 0.45) -> np.ndarray:
    """Rooms (coloured fill + bbox + label), doors (orange), windows (blue) over the original image."""
    h, w = maps.rooms.shape
    if image_bgr is None:
        base = np.full((h, w, 3), 255, np.uint8)
    elif image_bgr.shape[:2] != (h, w):
        base = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
    else:
        base = image_bgr.copy()
    layer = base.copy()
    for r in plan.rooms:
        layer[maps.rooms == r.index] = PALETTE_BGR[(r.index - 1) % len(PALETTE_BGR)]
    layer[maps.doors > 0] = DOOR_BGR
    layer[maps.windows > 0] = WINDOW_BGR
    out = cv2.addWeighted(layer, alpha, base, 1 - alpha, 0)
    out[maps.doors > 0] = DOOR_BGR
    out[maps.windows > 0] = WINDOW_BGR
    for r in plan.rooms:
        x, y, bw, bh = r.bbox
        cv2.rectangle(out, (x, y), (x + bw - 1, y + bh - 1), PALETTE_BGR[(r.index - 1) % len(PALETTE_BGR)], 2)
        _text(out, f"R{r.index} {r.label or 'room'}", (x + 4, y + 16))
        _text(out, f"{r.area_px} px", (x + 4, y + 32), 0.4)
    _text(out, f"{len(plan.rooms)} rooms, {len(plan.doors)} doors, {len(plan.windows)} windows", (4, 10), 0.33)
    if plan.is_synthetic:
        _text(out, "SYNTHETIC TEST DATA", (4, h - 3), 0.33)
    return out


def save_overlay(path: Path | str, image_bgr, plan, maps) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), render_overlay(image_bgr, plan, maps)):
        raise OSError(f"could not write overlay to {path}")
    return path


def save_comparison_figure(path: Path | str, image_bgr, overlay_bgr, title: str) -> Path:
    """Original | overlay side by side, saved with Matplotlib (Agg backend, no display needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, im, t in zip(axes, (image_bgr, overlay_bgr), ("Original", "Extracted overlay")):
        if im is not None:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t); ax.axis("off")
    fig.suptitle(f"{title}  (rooms: coloured boxes, doors: orange, windows: blue; areas in pixels)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
