from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from gamebuilder.MB3_env import mariobros3_env


def _set_env_default_state(env, *, state: str) -> None:
    """Try to set the reset wrapper's default state inside env."""

    try:
        from wrapper.reset_by_death import ResetToDefaultStateByDeathWrapper
    except Exception:
        return

    cur = env
    while True:
        if isinstance(cur, ResetToDefaultStateByDeathWrapper):
            cur.default_state = str(state)
            cur._force_default_state_on_reset = True
            return
        nxt = getattr(cur, "env", None)
        if nxt is None:
            return
        cur = nxt


def _overlay_text(
    bgr: np.ndarray,
    *,
    text: str,
    org: tuple[int, int] = (8, 20),
) -> np.ndarray:
    try:
        cv2.putText(
            bgr,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    except Exception:
        pass
    return bgr


def _choose_fourcc() -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(fourcc_fn):
        fourcc_val = fourcc_fn(*"mp4v")
        return int(fourcc_val)  # type: ignore[arg-type]
    return int(cv2.VideoWriter.fourcc(*"mp4v"))


def record_replay_video(
    *,
    custom_data_root: str,
    out_path: str | Path,
    action_trace: list[int],
    state_label: str,
    fps: int,
    max_steps: Optional[int] = None,
    to_int: Callable[[object], int],
    verbose: bool = False,
) -> tuple[Path, Optional[int]]:
    """Replay an action trace in a fresh env and record it to an MP4.

    Returns (final_path, achieved_max_x).
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a unique temp file first for best-effort atomic rename.
    # (Avoid collisions across processes.)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=str(out_path.parent),
        prefix=out_path.name + ".tmp.",
        suffix=out_path.suffix or ".mp4",
    ) as tmp:
        tmp_path = Path(tmp.name)

    from wrapper.death_position_logger import DeathPositionLoggerWrapper

    tmp_deaths_dir = tempfile.mkdtemp(prefix="video_eval_deaths_")
    tmp_log_path = str(Path(tmp_deaths_dir) / "deaths_eval.jsonl")

    base_env = DeathPositionLoggerWrapper(
        mariobros3_env(
            str(custom_data_root),
            rank=0,
            run_dir=None,
            enable_death_logger=False,
            render_mode="rgb_array",
        ),
        log_path=tmp_log_path,
    )

    # Example: "World1.Level3" -> "1Player.World1.Level3".
    if (
        state_label
        and state_label != "UnknownState"
        and not state_label.startswith("1Player.")
    ):
        replay_state = f"1Player.{state_label}"
    else:
        replay_state = state_label

    if replay_state and replay_state != "UnknownState":
        _set_env_default_state(base_env, state=replay_state)

    achieved_max_x: Optional[int] = None
    last_x: Optional[int] = None

    try:
        _obs, _info = base_env.reset()

        frame = base_env.render()
        if frame is None:
            raise RuntimeError(
                "Env render() returned None; cannot record video"
            )

        frame_arr = np.asarray(frame)
        if frame_arr.ndim != 3 or frame_arr.shape[2] != 3:
            raise RuntimeError(
                f"Unexpected render frame shape: {frame_arr.shape}"
            )

        height, width = int(frame_arr.shape[0]), int(frame_arr.shape[1])
        writer = cv2.VideoWriter(
            str(tmp_path),
            _choose_fourcc(),
            float(int(fps)),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to open cv2.VideoWriter")

        try:
            first_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
            first_bgr = _overlay_text(first_bgr, text="x=?")
            writer.write(first_bgr)

            limit = None
            if max_steps is not None:
                try:
                    limit = max(0, int(max_steps))
                except Exception:
                    limit = None

            for i, act in enumerate(action_trace):
                if limit is not None and i >= limit:
                    break
                _obs, _reward, terminated, truncated, info = base_env.step(
                    int(act)
                )
                ended = bool(terminated or truncated)

                # Track x during the replay.
                try:
                    x_val = info.get("x") if isinstance(info, dict) else None
                    if x_val is not None:
                        cur_x = int(to_int(x_val))
                        last_x = int(cur_x)
                        achieved_max_x = (
                            cur_x
                            if achieved_max_x is None
                            else max(int(achieved_max_x), int(cur_x))
                        )
                except Exception:
                    pass

                frame2 = base_env.render()
                if frame2 is not None:
                    arr2 = np.asarray(frame2)
                    if arr2.ndim == 3 and arr2.shape[2] == 3:
                        bgr2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2BGR)
                        if last_x is not None:
                            bgr2 = _overlay_text(
                                bgr2, text=f"x={int(last_x)}"
                            )
                        else:
                            bgr2 = _overlay_text(bgr2, text="x=?")
                        writer.write(bgr2)

                if ended:
                    break
        finally:
            writer.release()
    finally:
        base_env.close()

    # Best-effort rename.
    try:
        if tmp_path.exists():
            tmp_path.replace(out_path)
    except Exception:
        out_path = tmp_path

    if verbose:
        print(
            "[video-replay] wrote:",
            str(out_path),
            "recorded_max_x=",
            int(achieved_max_x) if achieved_max_x is not None else None,
        )

    return out_path, achieved_max_x
