"""Dump exactly what the agent observes (after preprocessing + frame stack).

This is useful because the normal Retro render window shows the raw game frame,
not the resized/grayscale/stacked observation actually fed into the policy.

Usage:
  ./project/bin/python -m utils.agent_view --out outputs/agent_view

It will write:
  - agent_view_last.png  (the most recent frame from the stack)
  - agent_view_stack.png (a horizontal montage of the stacked frames)

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecMonitor,
)

from gamebuilder.MB3_env import mariobros3_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump the processed observation (grayscale+resize+frame-stack) "
            "to PNGs so you can screenshot/verify what the agent sees."
        )
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/agent_view",
        help="Output directory for PNGs.",
    )
    parser.add_argument(
        "--n-stack",
        type=int,
        default=4,
        help="Number of frames stacked by VecFrameStack.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help=(
            "Number of random steps before dumping (helps fill the stack with "
            "different frames; after reset the stack is usually repeated)."
        ),
    )
    parser.add_argument(
        "--render-raw",
        action="store_true",
        help="Also call env.render() during warmup (shows raw retro window).",
    )
    parser.add_argument(
        "--custom-data-root",
        type=str,
        default=None,
        help=(
            "Path to retro_custom. Defaults to <repo>/retro_custom "
            "when omitted."
        ),
    )
    return parser.parse_args()


def _save_gray_png(arr_2d: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr_2d.astype(np.uint8), mode="L")
    img.save(str(path))


def _dump_stack(obs_0: np.ndarray, out_dir: Path) -> None:
    """Dump stack montage + last frame.

    Handles both channels-last (H,W,C) and channels-first (C,H,W).
    """

    arr = np.asarray(obs_0)

    if arr.ndim == 3 and arr.shape[-1] in (1, 2, 3, 4, 5, 6, 8, 12, 16):
        # channels-last
        frames = [arr[:, :, i] for i in range(arr.shape[-1])]
    elif arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4, 5, 6, 8, 12, 16):
        # channels-first
        frames = [arr[i, :, :] for i in range(arr.shape[0])]
    else:
        raise ValueError(
            "Unexpected stacked observation shape: "
            f"{arr.shape} (ndim={arr.ndim})"
        )

    frames_u8 = [np.asarray(f, dtype=np.uint8) for f in frames]

    # Last frame
    _save_gray_png(frames_u8[-1], out_dir / "agent_view_last.png")

    # Horizontal montage
    h, w = frames_u8[0].shape
    montage = Image.new("L", (w * len(frames_u8), h))
    for i, frame in enumerate(frames_u8):
        montage.paste(Image.fromarray(frame, mode="L"), (i * w, 0))
    montage.save(str(out_dir / "agent_view_stack.png"))


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    custom_data_root = (
        Path(args.custom_data_root).expanduser().resolve()
        if args.custom_data_root
        else (repo_root / "retro_custom")
    )

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([lambda: mariobros3_env(str(custom_data_root), rank=0)])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=int(args.n_stack))

    obs = env.reset()

    for _ in range(int(args.warmup_steps)):
        action = np.asarray([env.action_space.sample()])
        obs, _rewards, dones, _infos = env.step(action)
        if args.render_raw:
            env.render()
        if bool(np.any(dones)):
            obs = env.reset()

    obs0 = np.asarray(obs)[0]

    print(
        "[agent-view] obs shape:",
        tuple(np.asarray(obs).shape),
        "dtype=",
        np.asarray(obs).dtype,
        "min=",
        int(np.min(obs0)),
        "max=",
        int(np.max(obs0)),
    )

    _dump_stack(obs0, out_dir)
    print(f"[agent-view] wrote: {out_dir / 'agent_view_last.png'}")
    print(f"[agent-view] wrote: {out_dir / 'agent_view_stack.png'}")

    env.close()


if __name__ == "__main__":
    main()
