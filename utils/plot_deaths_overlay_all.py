import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _scale_x_to_image_width(
    xs: np.ndarray,
    width: int,
    *,
    percentile_ref: float = 99.5,
    already_scaled_slack: float = 1.05,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Map logged x coordinates to image pixel coordinates.

    The logged x values are in a game/world coordinate system. Depending on
    how the stitched level image was created, its pixel width may differ.
    We therefore scale x positions to fit the image width.

    Returns:
      x_pix: float64 array (same length as xs after masking)
      mask: boolean mask of finite entries applied to xs
      meta: dict with scaling info
    """

    width_i = int(width)
    xs_f = xs.astype(np.float64, copy=False)
    finite = np.isfinite(xs_f)
    xs_f = xs_f[finite]

    if xs_f.size == 0:
        return xs_f, finite, {"scaled": False, "scale_x": 1.0, "x_ref": None}

    x_min = float(xs_f.min())
    x_max = float(xs_f.max())
    x_ref = float(np.percentile(xs_f, percentile_ref))
    if not np.isfinite(x_ref) or x_ref <= 0:
        x_ref = x_max if x_max > 0 else 1.0

    # If coordinates already match image pixels (within slack), avoid scaling.
    if x_min >= 0 and x_ref <= (width_i - 1) * already_scaled_slack:
        x_pix = np.clip(xs_f, 0.0, float(width_i - 1))
        return x_pix, finite, {
            "scaled": False,
            "scale_x": 1.0,
            "x_ref": x_ref,
            "x_max": x_max,
        }

    scale_x = float(width_i - 1) / float(x_ref)
    x_pix = xs_f * scale_x
    # Still clip extreme outliers after scaling.
    x_pix = np.clip(x_pix, 0.0, float(width_i - 1))
    return x_pix, finite, {
        "scaled": True,
        "scale_x": scale_x,
        "x_ref": x_ref,
        "x_max": x_max,
    }


def _read_jsonl_x(path: Path) -> List[int]:
    xs: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            x_val = obj.get("x")
            if x_val is None:
                continue
            xs.append(int(x_val))
    return xs


def _load_deaths_from_dir(
    deaths_dir: Path,
) -> np.ndarray:
    if not deaths_dir.exists() or not deaths_dir.is_dir():
        raise FileNotFoundError(
            f"Deaths dir not found: {deaths_dir}. "
            "Run training first to generate outputs/deaths/*.jsonl."
        )

    paths = sorted(deaths_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No .jsonl files found in: {deaths_dir} "
            "(expected deaths_env*.jsonl)"
        )

    all_xs: List[int] = []
    for p in paths:
        all_xs.extend(_read_jsonl_x(p))

    return np.asarray(all_xs, dtype=np.int64)


def render_deaths_overlay_all(
    *,
    image: Path = Path("assets/level_1-1.png"),
    deaths_dir: Path = Path("outputs/deaths"),
    out: Path = Path("outputs/deaths_overlay_all.png"),
    default_y: Optional[int] = None,
    alpha: float = 0.25,
    size: float = 20.0,
) -> Path:
    # Always render onto the original sharp base image.
    base_image = Path("assets/level_1-1.png")
    img = Image.open(base_image).convert("RGBA")
    width, height = img.size

    xs = _load_deaths_from_dir(deaths_dir)
    if xs.size == 0:
        raise RuntimeError(f"No death entries found in: {deaths_dir}")

    # Scale x coordinates to the image width (robustly).
    x_pix, mask, meta = _scale_x_to_image_width(xs, width)
    if meta.get("scaled"):
        print(
            "[death-overlay] Scaling x to image width ("
            f"scale_x={meta['scale_x']:.4f}, "
            f"x_ref~p99.5={meta['x_ref']:.1f}, "
            f"x_max={meta['x_max']:.1f}, "
            f"width={width})"
        )
    xs = x_pix

    # Use a fixed y (65% of image height by default) so the overlay is stable.
    if default_y is None:
        default_y = int(height * 0.65)
    ys = np.full(xs.shape, fill_value=float(default_y), dtype=np.float64)

    out.parent.mkdir(parents=True, exist_ok=True)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    a = int(max(0.0, min(1.0, float(alpha))) * 255)
    color = (255, 0, 0, a)

    half_len = max(2, int(round(float(size) / 2.0)))
    stroke = 2
    for x_f, y_f in zip(xs, ys):
        if not np.isfinite(x_f) or not np.isfinite(y_f):
            continue
        x = int(round(float(x_f)))
        y = int(round(float(y_f)))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        draw.line(
            [(x - half_len, y - half_len), (x + half_len, y + half_len)],
            fill=color,
            width=stroke,
        )
        draw.line(
            [(x - half_len, y + half_len), (x + half_len, y - half_len)],
            fill=color,
            width=stroke,
        )

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    composed.save(out)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay ALL logged deaths on the level image.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("assets/level_1-1.png"),
        help="Path to the level image (PNG).",
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
        default=0.25,
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
        image=args.image,
        deaths_dir=args.deaths_dir,
        out=args.out,
        default_y=args.default_y,
        alpha=args.alpha,
        size=args.size,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
