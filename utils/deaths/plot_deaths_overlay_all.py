import argparse
from pathlib import Path
from typing import Optional

from utils.deaths.death_overlay import load_deaths_from_dir, render_overlay


def render_deaths_overlay_all(
    *,
    deaths_dir: Path = Path("outputs/deaths"),
    out: Path = Path("outputs/deaths_overlay_all.png"),
    default_y: Optional[int] = None,
    alpha: float = 0.35,
    size: float = 20.0,
) -> Path:
    xs = load_deaths_from_dir(deaths_dir)
    if xs.size == 0:
        raise RuntimeError(f"No death entries found in: {deaths_dir}")

    return render_overlay(
        image_path=Path("assets/level_1-1.png"),
        xs=xs,
        ys=None,
        out=out,
        default_y=default_y,
        alpha=alpha,
        size=size,
    )


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
