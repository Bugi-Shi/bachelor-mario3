from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def scale_x_to_image_width(
    xs: np.ndarray,
    width: int,
    *,
    percentile_ref: float = 99.5,
    already_scaled_slack: float = 1.05,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Scale global-x values into image pixel coordinates.

    Returns:
        x_pix: scaled (or passthrough) x values for finite entries only
        mask: boolean mask indicating which original xs entries were finite
        meta: debug metadata
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

    # If values are already roughly in [0, width), keep them.
    if x_min >= 0 and x_ref <= (width_i - 1) * already_scaled_slack:
        x_pix = np.clip(xs_f, 0.0, float(width_i - 1))
        return x_pix, finite, {
            "scaled": False,
            "scale_x": 1.0,
            "x_ref": x_ref,
            "x_max": x_max,
        }

    scale_x = float(width_i - 1) / float(x_ref)
    x_pix = np.clip(xs_f * scale_x, 0.0, float(width_i - 1))
    return x_pix, finite, {
        "scaled": True,
        "scale_x": scale_x,
        "x_ref": x_ref,
        "x_max": x_max,
    }


def draw_crosses(
    img_rgba: Image.Image,
    xs: Iterable[Any],
    ys: Iterable[Any],
    *,
    alpha: float,
    size: float,
    color_rgb: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    img = img_rgba.convert("RGBA")
    width, height = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    a = int(max(0.0, min(1.0, float(alpha))) * 255)
    r, g, b = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))
    color = (r, g, b, a)

    half_len = max(2, int(round(float(size) / 2.0)))
    stroke = 2

    for x_f, y_f in zip(xs, ys):
        try:
            x = int(round(float(x_f)))
            y = int(round(float(y_f)))
        except Exception:
            continue
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

    return Image.alpha_composite(img, overlay)


def render_overlay_multi(
    *,
    image_path: Path,
    groups: Sequence[
        Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]
    ],
    out: Path,
    default_y: Optional[int] = None,
    alpha: float = 0.6,
    size: float = 20.0,
    print_scale_meta: bool = True,
) -> Path:
    """Render an overlay with multiple marker groups.

    Intended for distinguishing e.g. death reasons by color.
    """

    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    half_len = max(2, int(round(float(size) / 2.0)))
    if default_y is None:
        # Place markers near the bottom edge but keep them fully visible.
        default_y = max(0, int(height - 1 - (half_len + 2)))

    for xs, ys, color_rgb in groups:
        if xs is None:
            continue
        xs_arr = np.asarray(xs, dtype=np.float64)
        if xs_arr.size == 0:
            continue

        x_pix, mask, meta = scale_x_to_image_width(xs_arr, width)
        if print_scale_meta and meta.get("scaled"):
            print(
                "[death-overlay] Scaling x to image width ("
                f"scale_x={meta['scale_x']:.4f}, "
                f"x_ref~p99.5={meta['x_ref']:.1f}, "
                f"x_max={meta['x_max']:.1f}, "
                f"width={width})"
            )

        ys_pix: np.ndarray
        if ys is not None:
            ys_arr = np.asarray(ys, dtype=np.int64)
            ys_arr = ys_arr[mask]
            ys_pix = np.clip(ys_arr, 0, height - 1)
        else:
            ys_pix = np.full(
                x_pix.shape,
                fill_value=int(default_y),
                dtype=np.int64,
            )

        img = draw_crosses(
            img,
            x_pix,
            ys_pix,
            alpha=alpha,
            size=size,
            color_rgb=color_rgb,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out)
    return out


def read_jsonl_x(path: Path) -> List[int]:
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
            x_val = obj.get("x")
            if x_val is None:
                continue
            xs.append(int(x_val))
    return xs


def load_deaths_from_dir(deaths_dir: Path) -> np.ndarray:
    if not deaths_dir.exists() or not deaths_dir.is_dir():
        raise FileNotFoundError(
            f"Deaths dir not found: {deaths_dir}. "
            "Run training first to generate deaths_env*.jsonl."
        )

    paths = sorted(deaths_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No .jsonl files found in: {deaths_dir} "
            "(expected deaths_env*.jsonl)"
        )

    all_xs: List[int] = []
    for p in paths:
        all_xs.extend(read_jsonl_x(p))

    return np.asarray(all_xs, dtype=np.int64)


def load_deaths_file(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load death positions.

    Supported formats:
    - .npy: shape (N,) x-values or shape (N,2) (x,y)
    - .jsonl: each line has at least {"x": int} and optionally {"y": int}
    - .json: list of ints (x) OR list of dicts with keys x and optional y
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
            ys = [
                int(o["y"])
                for o in obj
                if "y" in o and o["y"] is not None
            ]
            has_y = len(ys) == len(xs)
            return (
                np.asarray(xs, dtype=np.int64),
                np.asarray(ys, dtype=np.int64) if has_y else None,
            )
        raise ValueError("Unsupported .json format")

    raise ValueError(f"Unsupported file extension: {path.suffix}")


def render_overlay(
    *,
    image_path: Path,
    xs: np.ndarray,
    ys: Optional[np.ndarray] = None,
    out: Path,
    default_y: Optional[int] = None,
    alpha: float = 0.6,
    size: float = 20.0,
    print_scale_meta: bool = True,
    color_rgb: Tuple[int, int, int] = (255, 0, 0),
) -> Path:
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    x_pix, mask, meta = scale_x_to_image_width(xs, width)
    if print_scale_meta and meta.get("scaled"):
        print(
            "[death-overlay] Scaling x to image width ("
            f"scale_x={meta['scale_x']:.4f}, "
            f"x_ref~p99.5={meta['x_ref']:.1f}, "
            f"x_max={meta['x_max']:.1f}, "
            f"width={width})"
        )

    if ys is not None:
        ys = ys[mask]

    half_len = max(2, int(round(float(size) / 2.0)))

    if ys is None:
        y0 = default_y
        if y0 is None:
            # Place markers near the bottom edge but keep them fully visible.
            y0 = max(0, int(height - 1 - (half_len + 2)))
        ys = np.full(x_pix.shape, fill_value=int(y0), dtype=np.int64)
    else:
        ys = np.clip(ys, 0, height - 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    composed = draw_crosses(
        img,
        x_pix,
        ys,
        alpha=alpha,
        size=size,
        color_rgb=color_rgb,
    )
    composed.convert("RGB").save(out)
    return out
