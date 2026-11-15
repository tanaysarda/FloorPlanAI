"""Robust image/mask loading and the dataset-adapter interface.

To integrate the real Kaggle dataset: inspect its layout, then either point FolderPairAdapter at the
image/mask folders or write a small DatasetAdapter subclass yielding Sample objects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class LoadError(Exception):
    """A file is missing, unreadable or corrupt. Messages are safe to display."""


@dataclass(frozen=True)
class Sample:
    name: str
    image_path: Optional[Path]
    mask_path: Optional[Path]


def _read(path: Path | str, flag: int, what: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise LoadError(f"{what} file not found: {p.name}")
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        arr = cv2.imdecode(data, flag) if data.size else None
    except (OSError, cv2.error) as exc:
        raise LoadError(f"{what} file could not be read: {p.name}") from exc
    if arr is None:
        raise LoadError(f"{what} file is corrupt or not a supported image: {p.name}")
    return arr


def load_image(path: Path | str) -> np.ndarray:
    """Load a BGR colour image."""
    return _read(path, cv2.IMREAD_COLOR, "image")


def load_mask_raw(path: Path | str) -> np.ndarray:
    """Load a mask unchanged (no interpolation, no colour conversion)."""
    return _read(path, cv2.IMREAD_UNCHANGED, "mask")


class DatasetAdapter:
    """Yield Samples for a dataset. Subclass or configure for the real dataset."""
    dataset_name: str = "unnamed"
    is_synthetic: bool = False

    def samples(self) -> List[Sample]:  # pragma: no cover - interface
        raise NotImplementedError


class FolderPairAdapter(DatasetAdapter):
    """Pairs images and masks by file stem: images/<stem>.<ext> <-> masks/<stem><mask_suffix>.<ext>."""

    def __init__(self, image_dir: Path | str, mask_dir: Path | str, dataset_name: str,
                 mask_suffix: str = "", is_synthetic: bool = False, image_exts: Sequence[str] = IMAGE_EXTS):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.dataset_name, self.mask_suffix, self.is_synthetic = dataset_name, mask_suffix, is_synthetic
        self.image_exts = tuple(e.lower() for e in image_exts)

    def _find_mask(self, stem: str) -> Optional[Path]:
        for ext in IMAGE_EXTS:
            cand = self.mask_dir / f"{stem}{self.mask_suffix}{ext}"
            if cand.is_file():
                return cand
        return None

    def samples(self) -> List[Sample]:
        if not self.image_dir.is_dir():
            return []
        out = []
        for img in sorted(self.image_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in self.image_exts:
                out.append(Sample(img.stem, img, self._find_mask(img.stem)))
        return out
