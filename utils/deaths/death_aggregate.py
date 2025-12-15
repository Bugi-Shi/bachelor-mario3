from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
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


def _normalize_global_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a global allDeath.jsonl entry to the new schema.

    Required keys:
      - ep: int (0 if unknown)
      - reason: str | None
      - x: int | None
      - level: str ("Unknown" if unknown)
      - source: str
    """

    ep_raw = obj.get("ep", obj.get("episode", obj.get("episode_idx")))
    try:
        ep = int(ep_raw) if ep_raw is not None else 0
    except Exception:
        ep = 0

    x_raw = obj.get("x")
    try:
        x = int(x_raw) if x_raw is not None else None
    except Exception:
        x = None

    level_raw = obj.get("level")
    level = (
        level_raw
        if isinstance(level_raw, str) and level_raw
        else "Unknown"
    )

    reason_raw = obj.get("reason")
    reason = reason_raw if isinstance(reason_raw, str) else None

    source_raw = obj.get("source")
    source = source_raw if isinstance(source_raw, str) else ""

    return {
        "ep": ep,
        "reason": reason,
        "x": x,
        "level": level,
        "source": source,
    }


def migrate_global_jsonl_schema(
    *,
    global_jsonl: Path = _GLOBAL_JSONL,
) -> bool:
    """Rewrite outputs/allDeath.jsonl to the new schema.

    Returns True if a rewrite happened.
    """

    global_jsonl = Path(global_jsonl)
    if not global_jsonl.exists():
        return False

    needs_rewrite = False
    with global_jsonl.open("r", encoding="utf-8") as f:
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
            # Old schema used run/env/hpos/... and lacked ep/level.
            if (
                "ep" not in obj
                or "level" not in obj
                or "run" in obj
                or "env" in obj
            ):
                needs_rewrite = True
                break

    if not needs_rewrite:
        return False

    global_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(global_jsonl.parent),
        prefix=global_jsonl.name + ".tmp.",
    ) as tmp:
        tmp_path = Path(tmp.name)
        with global_jsonl.open("r", encoding="utf-8") as f:
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
                normalized = _normalize_global_entry(obj)
                # If the old entry lacked source, idempotency can't be kept;
                # skip it.
                if not normalized.get("source"):
                    continue
                tmp.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    tmp_path.replace(global_jsonl)
    return True


def append_run_deaths_to_global(
    *,
    run_dir: Path,
    global_jsonl: Path = _GLOBAL_JSONL,
) -> int:
    """Append deaths from a run into outputs/allDeath.jsonl (idempotent).

    Each entry contains:
            - ep, reason, x, level
      - source: unique source pointer (run/<file>#L<line>)
    """

    run_dir = Path(run_dir)
    deaths_dir = run_dir / "deaths"
    if not deaths_dir.exists() or not deaths_dir.is_dir():
        return 0

    global_jsonl = Path(global_jsonl)
    global_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Keep outputs/allDeath.jsonl consistent if it exists in the old schema.
    migrate_global_jsonl_schema(global_jsonl=global_jsonl)

    seen = _iter_global_sources_jsonl(global_jsonl)

    added = 0
    for jsonl_path in sorted(deaths_dir.glob("*.jsonl")):
        for line_no, obj in _read_jsonl(jsonl_path):
            source = f"{run_dir.name}/{jsonl_path.name}#L{line_no}"
            if source in seen:
                continue

            # Run death logs should be {ep, reason, x, level};
            # tolerate older logs.
            ep_raw = obj.get("ep")
            try:
                ep = int(ep_raw) if ep_raw is not None else 0
            except Exception:
                ep = 0

            level_raw = obj.get("level")
            level = (
                level_raw
                if isinstance(level_raw, str) and level_raw
                else "Unknown"
            )

            x_raw = obj.get("x")
            try:
                x = int(x_raw) if x_raw is not None else None
            except Exception:
                x = None

            reason_raw = obj.get("reason")
            reason = reason_raw if isinstance(reason_raw, str) else None

            entry: Dict[str, Any] = {
                "ep": ep,
                "reason": reason,
                "x": x,
                "level": level,
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
