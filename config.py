"""Central configuration: paths, thresholds and dependency-free .env loading."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CONFIG_DIR = ROOT / "configs"
SCHEMA_PATH = ROOT / "database" / "schema.sql"
SYNTHETIC_DIR = DATA_DIR / "synthetic"      # generated TEST data, never Kaggle data
REAL_DATA_DIR = DATA_DIR / "kaggle"         # put the real dataset here (git-ignored)

DATASET_KINDS = ("synthetic", "real")
QUERY_TIMEOUT_S = 2.0
QUERY_MAX_ROWS = 200


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing vars."""
    path = path or ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Paths:
    kind: str
    output_dir: Path
    db_path: Path
    overlays_dir: Path
    json_path: Path
    report_path: Path


def paths_for(kind: str, output_root: Path | None = None) -> Paths:
    """Return output locations. Synthetic and real results are always kept apart."""
    if kind not in DATASET_KINDS:
        raise ValueError(f"dataset kind must be one of {DATASET_KINDS}, got {kind!r}")
    out = (output_root or OUTPUT_DIR) / kind
    return Paths(kind, out, out / "floorplans.db", out / "overlays",
                 out / "floorplans.json", out / "ingest_report.json")


def rel_path(p: Path | str) -> str:
    """Store project-relative paths when possible so the DB is portable."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_path(p: str | None) -> Path | None:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else ROOT / q
