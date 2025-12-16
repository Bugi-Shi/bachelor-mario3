from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def create_run_dir(
    *,
    base_dir: Path = Path("outputs") / "runs",
    time_format: str = "%Y-%m-%d_%H-%M-%S",
) -> Path:
    """Create and return a unique run directory.

    Default format is lexicographically sortable, so the latest run is the max.
    If a directory already exists (e.g. two runs in the same second), a suffix
    _2, _3, ... is appended.
    """

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime(time_format)
    candidate = base_dir / run_id
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    # Collision within the same second -> add suffix.
    idx = 2
    while True:
        candidate = base_dir / f"{run_id}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def latest_run_dir(
    *, base_dir: Path = Path("outputs") / "runs"
) -> Optional[Path]:
    base_dir = Path(base_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        return None

    runs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not runs:
        return None
    return runs[-1]
