from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from utils.deaths.death_overlay import render_overlay_multi


_COLOR_LIFE_LOST = (255, 0, 0)
_COLOR_STUCK = (255, 165, 0)


def _asset_for_level(level: str) -> Path:
    """Map a short level code to the corresponding background image."""

    lvl = str(level).strip()
    if not lvl or lvl == "Unknown":
        # Default to 1-1.
        lvl = "1-1"

    # World 1.
    if lvl == "1-A":
        return Path("assets") / "SuperMarioBros3Map1Airship.png"
    if lvl == "1-MF":
        return Path("assets") / "SuperMarioBros3Map1MiniFortress.png"

    # Regular levels like 1-1..1-6.
    return Path("assets") / f"SuperMarioBros3Map{lvl}.png"


def _load_deaths_grouped(deaths_dir: Path) -> dict[str, dict[str, list[int]]]:
    grouped: dict[str, dict[str, list[int]]] = {}
    for p in sorted(Path(deaths_dir).glob("*.jsonl")):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
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
                x_i = int(x)
            except Exception:
                continue
            level = obj.get("level")
            level_s = str(level).strip() if level is not None else "Unknown"
            reason = obj.get("reason")
            reason_s = (
                str(reason).strip()
                if reason is not None
                else "life_lost"
            )
            grouped.setdefault(level_s, {}).setdefault(
                reason_s, []
            ).append(x_i)
    return grouped


def _load_deaths_grouped_from_jsonl(
    path: Path,
) -> dict[str, dict[str, list[int]]]:
    grouped: dict[str, dict[str, list[int]]] = {}
    p = Path(path)
    if not p.exists() or not p.is_file():
        return grouped

    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return grouped

    for line in text.splitlines():
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
            x_i = int(x)
        except Exception:
            continue
        level = obj.get("level")
        level_s = str(level).strip() if level is not None else "Unknown"
        reason = obj.get("reason")
        reason_s = (
            str(reason).strip()
            if reason is not None
            else "life_lost"
        )
        grouped.setdefault(level_s, {}).setdefault(reason_s, []).append(x_i)

    return grouped


def render_deaths_overlay_all(
    *,
    deaths_dir: Path = Path("outputs/deaths"),
    out: Path = Path("outputs/deaths_overlay_all.png"),
    default_y: Optional[int] = None,
    alpha: float = 0.6,
    size: float = 20.0,
) -> Path:
    deaths_dir = Path(deaths_dir)
    grouped = _load_deaths_grouped(deaths_dir)
    if not grouped:
        raise RuntimeError(f"No death entries found in: {deaths_dir}")

    out = Path(out)
    wrote: list[Path] = []

    def _groups_for(level_bucket: dict[str, list[int]]):
        life = np.asarray(level_bucket.get("life_lost", []), dtype=np.int64)
        stuck = np.asarray(level_bucket.get("stuck", []), dtype=np.int64)
        other: list[int] = []
        for k, v in level_bucket.items():
            if k in {"life_lost", "stuck"}:
                continue
            other.extend(v)
        other_arr = np.asarray(other, dtype=np.int64)

        groups = []
        if life.size:
            groups.append((life, None, _COLOR_LIFE_LOST))
        if stuck.size:
            groups.append((stuck, None, _COLOR_STUCK))
        if other_arr.size:
            groups.append((other_arr, None, _COLOR_LIFE_LOST))
        return groups

    # If only one level exists, keep the caller-specified output name.
    if len(grouped) == 1:
        (level, bucket) = next(iter(grouped.items()))
        image_path = _asset_for_level(level)
        return render_overlay_multi(
            image_path=image_path,
            groups=_groups_for(bucket),
            out=out,
            default_y=default_y,
            alpha=alpha,
            size=size,
        )

    # Multiple levels: write one overlay per level.
    for level, bucket in sorted(grouped.items()):
        if not bucket:
            continue
        image_path = _asset_for_level(level)
        out_level = out.with_name(f"{out.stem}_{level}{out.suffix}")
        wrote.append(
            render_overlay_multi(
                image_path=image_path,
                groups=_groups_for(bucket),
                out=out_level,
                default_y=default_y,
                alpha=alpha,
                size=size,
            )
        )

    return wrote[0] if wrote else out


def render_deaths_overlay_from_jsonl(
    *,
    deaths_jsonl: Path,
    out: Path,
    default_y: Optional[int] = None,
    alpha: float = 0.6,
    size: float = 20.0,
) -> Path:
    """Render one (or multiple) overlays grouped by level from a JSONL file.

    Expected schema per line: at least {"x": int, "level": str}.
    """

    grouped = _load_deaths_grouped_from_jsonl(Path(deaths_jsonl))
    if not grouped:
        raise RuntimeError(f"No death entries found in: {deaths_jsonl}")

    out = Path(out)
    wrote: list[Path] = []

    def _groups_for(level_bucket: dict[str, list[int]]):
        life = np.asarray(level_bucket.get("life_lost", []), dtype=np.int64)
        stuck = np.asarray(level_bucket.get("stuck", []), dtype=np.int64)
        other: list[int] = []
        for k, v in level_bucket.items():
            if k in {"life_lost", "stuck"}:
                continue
            other.extend(v)
        other_arr = np.asarray(other, dtype=np.int64)

        groups = []
        if life.size:
            groups.append((life, None, _COLOR_LIFE_LOST))
        if stuck.size:
            groups.append((stuck, None, _COLOR_STUCK))
        if other_arr.size:
            groups.append((other_arr, None, _COLOR_LIFE_LOST))
        return groups

    if len(grouped) == 1:
        (level, bucket) = next(iter(grouped.items()))
        image_path = _asset_for_level(level)
        return render_overlay_multi(
            image_path=image_path,
            groups=_groups_for(bucket),
            out=out,
            default_y=default_y,
            alpha=alpha,
            size=size,
        )

    for level, bucket in sorted(grouped.items()):
        if not bucket:
            continue
        image_path = _asset_for_level(level)
        out_level = out.with_name(f"{out.stem}_{level}{out.suffix}")
        wrote.append(
            render_overlay_multi(
                image_path=image_path,
                groups=_groups_for(bucket),
                out=out_level,
                default_y=default_y,
                alpha=alpha,
                size=size,
            )
        )

    return wrote[0] if wrote else out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay ALL logged deaths on the level image.",
    )
    parser.add_argument(
        "--deaths-dir",
        type=Path,
        default=Path("outputs/deaths"),
        help="Directory containing deaths_env*.jsonl logs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/deaths_overlay_all.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--default-y",
        type=int,
        default=None,
        help=(
            "If logs contain only x, use this y pixel "
            "(default: near bottom edge)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Marker alpha.",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=20.0,
        help="Marker size.",
    )
    args = parser.parse_args()

    out = render_deaths_overlay_all(
        deaths_dir=args.deaths_dir,
        out=args.out,
        default_y=args.default_y,
        alpha=args.alpha,
        size=args.size,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
