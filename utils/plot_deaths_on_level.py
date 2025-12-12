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
    x_pix = np.clip(x_pix, 0.0, float(width_i - 1))
    return x_pix, finite, {
        "scaled": True,
        "scale_x": scale_x,
        "x_ref": x_ref,
        "x_max": x_max,
    }


def _load_deaths(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load death positions.

    Supported formats:
    - .npy: either shape (N,) of x-values or shape (N,2) of (x,y)
    - .jsonl: each line is a JSON object with keys {"x": int, "y": int?}
    - .json: list of ints (x) OR list of {"x":..,"y":..}
    """

    if not path.exists():
        raise FileNotFoundError(f"Deaths file not found: {path}")

    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            return arr.astype(np.int64), None
        if arr.ndim == 2 and arr.shape[1] == 2:
            return (arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64))
        raise ValueError(f"Unsupported .npy shape: {arr.shape}")

    if path.suffix == ".jsonl":
        xs: List[int] = []
        ys: List[int] = []
        has_y = True
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
                if "y" in obj and obj["y"] is not None:
                    ys.append(int(obj["y"]))
                else:
                    has_y = False
        x_arr = np.asarray(xs, dtype=np.int64)
        if has_y:
            return x_arr, np.asarray(ys, dtype=np.int64)
        return x_arr, None

    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list) and (
            len(obj) == 0 or isinstance(obj[0], int)
        ):
            return np.asarray(obj, dtype=np.int64), None
        if isinstance(obj, list) and (
            len(obj) == 0 or isinstance(obj[0], dict)
        ):
            xs = [int(o["x"]) for o in obj]
            ys = [int(o["y"]) for o in obj if "y" in o and o["y"] is not None]
            has_y = len(ys) == len(xs)
            return (
                np.asarray(xs, dtype=np.int64),
                np.asarray(ys, dtype=np.int64) if has_y else None,
            )
        raise ValueError("Unsupported .json format")

    raise ValueError(f"Unsupported file extension: {path.suffix}")


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
        default=0.25,
        help="Marker alpha (0..1).",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=20.0,
        help="Marker size (roughly cross diameter in pixels).",
    )
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGBA")
    width, height = img.size

    xs, ys = _load_deaths(args.deaths)
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
    if ys is not None:
        ys = ys[mask]

    if ys is None:
        default_y = args.default_y
        if default_y is None:
            default_y = int(height * 0.65)
        ys = np.full(xs.shape, fill_value=int(default_y), dtype=np.int64)
    else:
        ys = np.clip(ys, 0, height - 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    a = int(max(0.0, min(1.0, float(args.alpha))) * 255)
    color = (255, 0, 0, a)

    half_len = max(2, int(round(float(args.size) / 2.0)))
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
    composed.save(args.out)

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
