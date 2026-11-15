"""End-to-end processing: sample -> mask -> structured plan -> database (+ overlays)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import rel_path
from database import db
from processing.extraction import ExtractionParams, InstanceMaps, extract_plan
from processing.loader import DatasetAdapter, LoadError, Sample, load_image, load_mask_raw
from processing.models import FloorPlanData
from processing.segmentation import ClassMap, MaskFormatError
from processing.visualization import save_comparison_figure, save_overlay, render_overlay


@dataclass
class IngestReport:
    processed: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    overlays: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"processed": self.processed, "skipped": [{"name": n, "reason": r} for n, r in self.skipped],
                "overlays": self.overlays}


def process_sample(sample: Sample, class_map: ClassMap, dataset: str, is_synthetic: bool = False,
                   params: ExtractionParams = ExtractionParams()
                   ) -> Tuple[Optional[FloorPlanData], Optional[InstanceMaps], Optional[np.ndarray], Optional[str]]:
    """Returns (plan, instance_maps, image, error). Never raises for bad input; reports an error string."""
    if sample.mask_path is None:
        return None, None, None, "no segmentation mask found for this image"
    warnings: List[str] = []
    try:
        raw = load_mask_raw(sample.mask_path)
        class_ids, unknown = class_map.to_class_ids(raw)
    except (LoadError, MaskFormatError) as exc:
        return None, None, None, str(exc)
    image = None
    if sample.image_path is None:
        warnings.append("no original image; overlays use a blank background")
    else:
        try:
            image = load_image(sample.image_path)
            if image.shape[:2] != class_ids.shape:
                warnings.append(f"image size {image.shape[1]}x{image.shape[0]} differs from mask size "
                                f"{class_ids.shape[1]}x{class_ids.shape[0]}; image resized to the mask for overlays")
        except LoadError as exc:
            warnings.append(f"{exc}; overlay drawn on a blank background")
    plan, maps = extract_plan(
        sample.name, class_ids, class_map, dataset=dataset,
        image_path=rel_path(sample.image_path) if sample.image_path else None, mask_path=rel_path(sample.mask_path),
        is_synthetic=is_synthetic, unknown_fraction=unknown, params=params, warnings=warnings)
    return plan, maps, image, None


def ingest_dataset(adapter: DatasetAdapter, class_map: ClassMap, conn: sqlite3.Connection, *,
                   params: ExtractionParams = ExtractionParams(), overlays_dir: Optional[Path] = None,
                   max_overlays: Optional[int] = None, comparison_figures: bool = False) -> IngestReport:
    """Process every sample and store it. Bad samples are skipped and reported, never fatal. Idempotent."""
    report = IngestReport()
    for sample in sorted(adapter.samples(), key=lambda s: s.name):
        plan, maps, image, err = process_sample(sample, class_map, adapter.dataset_name, adapter.is_synthetic, params)
        if err:
            report.skipped.append((sample.name, err))
            continue
        try:
            db.upsert_floor_plan(conn, plan)
        except db.DatabaseError as exc:
            report.skipped.append((sample.name, str(exc)))
            continue
        report.processed.append(sample.name)
        if overlays_dir is not None and (max_overlays is None or len(report.overlays) < max_overlays):
            p = save_overlay(Path(overlays_dir) / f"{sample.name}_overlay.png", image, plan, maps)
            report.overlays.append(rel_path(p))
            if comparison_figures:
                save_comparison_figure(Path(overlays_dir) / f"{sample.name}_comparison.png", image,
                                       render_overlay(image, plan, maps), plan.name)
    return report


def write_report(report: IngestReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
