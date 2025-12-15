import argparse
from pathlib import Path

from utils.deaths.death_overlay import load_deaths_file, render_overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay death positions on a level image (cross markers)."
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("assets/level_1-1.png"),
        help="Path to the level image (PNG).",
    )
    parser.add_argument(
        "--deaths",
        type=Path,
        required=True,
        help="Deaths file (.npy/.json/.jsonl). x or (x,y) positions.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/deaths_overlay.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--default-y",
        type=int,
        default=None,
        help=(
            "If deaths only contain x, use this y pixel "
            "(default: ~65% of image height)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Marker alpha (0..1).",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=20.0,
        help="Marker size (roughly cross diameter in pixels).",
    )
    args = parser.parse_args()

    xs, ys = load_deaths_file(args.deaths)
    out = render_overlay(
        image_path=args.image,
        xs=xs,
        ys=ys,
        out=args.out,
        default_y=args.default_y,
        alpha=args.alpha,
        size=args.size,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
