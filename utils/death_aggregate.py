from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


_ENV_RE = re.compile(r"deaths_env(?P<env>\d+)\.jsonl$")
_GLOBAL_JSONL = Path("outputs") / "allDeath.jsonl"


def _parse_env_from_filename(path: Path) -> Optional[int]:
    m = _ENV_RE.search(path.name)
    if not m:
        return None
    try:
        return int(m.group("env"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield idx, obj


def _iter_global_sources_jsonl(path: Path) -> Set[str]:
    """Read existing sources from the global JSONL file for idempotency."""

    seen: Set[str] = set()
    if not path.exists():
        return seen

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            src = obj.get("source")
            if isinstance(src, str):
                seen.add(src)
    return seen


def append_run_deaths_to_global(
    *,
    run_dir: Path,
    global_jsonl: Path = _GLOBAL_JSONL,
) -> int:
    """Append deaths from a run into outputs/allDeath.jsonl (idempotent).

    Each entry contains:
      - run: run folder name
      - env: env index (if parseable)
      - reason, x, hpos, live_lost, stuck_lost, screen_idx
      - source: unique source pointer (run/<file>#L<line>)
    """

    run_dir = Path(run_dir)
    deaths_dir = run_dir / "deaths"
    if not deaths_dir.exists() or not deaths_dir.is_dir():
        return 0

    global_jsonl = Path(global_jsonl)
    global_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen = _iter_global_sources_jsonl(global_jsonl)

    added = 0
    for jsonl_path in sorted(deaths_dir.glob("*.jsonl")):
        env_idx = _parse_env_from_filename(jsonl_path)
        for line_no, obj in _read_jsonl(jsonl_path):
            source = f"{run_dir.name}/{jsonl_path.name}#L{line_no}"
            if source in seen:
                continue

            entry: Dict[str, Any] = {
                "run": run_dir.name,
                "env": env_idx,
                "reason": obj.get("reason"),
                "x": obj.get("x"),
                "hpos": obj.get("hpos"),
                "live_lost": obj.get("live_lost"),
                "stuck_lost": obj.get("stuck_lost"),
                "screen_idx": obj.get("screen_idx"),
                "source": source,
            }
            seen.add(source)
            added += 1

            # Append immediately (JSONL) to keep memory usage low.
            with global_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return added


def load_global_death_xs(
    *,
    global_jsonl: Path = _GLOBAL_JSONL,
) -> np.ndarray:
    """Load global x positions from outputs/allDeath.jsonl."""

    path = Path(global_jsonl)
    if not path.exists():
        return np.asarray([], dtype=np.int64)

    xs: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            x = obj.get("x")
            if x is None:
                continue
            try:
                xs.append(int(x))
            except Exception:
                continue

    return np.asarray(xs, dtype=np.int64)
