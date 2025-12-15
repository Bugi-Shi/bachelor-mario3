from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from utils.deaths.death_overlay import render_overlay


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


def _load_deaths_grouped(deaths_dir: Path) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
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
            grouped.setdefault(level_s, []).append(x_i)
    return grouped


def render_deaths_overlay_all(
    *,
    deaths_dir: Path = Path("outputs/deaths"),
    out: Path = Path("outputs/deaths_overlay_all.png"),
    default_y: Optional[int] = None,
    alpha: float = 0.35,
    size: float = 20.0,
) -> Path:
    deaths_dir = Path(deaths_dir)
    grouped = _load_deaths_grouped(deaths_dir)
    if not grouped:
        raise RuntimeError(f"No death entries found in: {deaths_dir}")

    out = Path(out)
    wrote: list[Path] = []

    # If only one level exists, keep the caller-specified output name.
    if len(grouped) == 1:
        (level, xs_list) = next(iter(grouped.items()))
        image_path = _asset_for_level(level)
        return render_overlay(
            image_path=image_path,
            xs=np.asarray(xs_list, dtype=np.int64),
            ys=None,
            out=out,
            default_y=default_y,
            alpha=alpha,
            size=size,
        )

    # Multiple levels: write one overlay per level.
    for level, xs_list in sorted(grouped.items()):
        if not xs_list:
            continue
        image_path = _asset_for_level(level)
        out_level = out.with_name(f"{out.stem}_{level}{out.suffix}")
        wrote.append(
            render_overlay(
                image_path=image_path,
                xs=np.asarray(xs_list, dtype=np.int64),
                ys=None,
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
            "(default: ~65% of image height)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
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
