from __future__ import annotations

import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from gamebuilder.MB3_env import mariobros3_env
from utils import to_python_int
from utils.pretty_terminal.ignore_signals import ignore_sigint


def _set_env_default_state(env: object, *, state: str) -> None:
    """Best-effort: set the reset wrapper's default state inside `env`."""

    try:
        from wrapper.reset_by_death import ResetToDefaultStateByDeathWrapper
    except Exception:
        return

    current = env
    while True:
        if isinstance(current, ResetToDefaultStateByDeathWrapper):
            current.default_state = str(state)
            current._force_default_state_on_reset = True
            return
        next_env = getattr(current, "env", None)
        if next_env is None:
            return
        current = next_env


def _overlay_text(
    frame_bgr: np.ndarray,
    *,
    text: str,
    org: tuple[int, int] = (8, 20),
) -> np.ndarray:
    try:
        cv2.putText(
            frame_bgr,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
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
    return frame_bgr


def _choose_fourcc() -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(fourcc_fn):
        fourcc_val = fourcc_fn(*"mp4v")
        return int(fourcc_val)  # type: ignore[arg-type]
    return int(cv2.VideoWriter.fourcc(*"mp4v"))


def _record_replay_video_impl(
    *,
    custom_data_root: str,
    out_path: str | Path,
    action_trace: list[int],
    state_label: str,
    fps: int,
    max_steps: Optional[int],
    to_int: Callable[[object], int],
    verbose: bool,
) -> tuple[Path, Optional[int]]:
    """In-process implementation for replay+record.

    Keep this separated so we can call it inside a subprocess worker without
    recursively spawning.
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
        tmp_video_path = Path(tmp.name)

    from wrapper.death_position_logger import DeathPositionLoggerWrapper

    achieved_max_x: Optional[int] = None
    last_x: Optional[int] = None

    with tempfile.TemporaryDirectory(prefix="video_eval_deaths_") as tmp_dir:
        tmp_log_path = str(Path(tmp_dir) / "deaths_eval.jsonl")
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

            height, width = (
                int(frame_arr.shape[0]),
                int(frame_arr.shape[1]),
            )
            writer = cv2.VideoWriter(
                str(tmp_video_path),
                _choose_fourcc(),
                float(int(fps)),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("Failed to open cv2.VideoWriter")

            try:
                first_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
                writer.write(_overlay_text(first_bgr, text="x=?"))

                step_limit: Optional[int] = None
                if max_steps is not None:
                    try:
                        step_limit = max(0, int(max_steps))
                    except Exception:
                        step_limit = None

                for step_i, action in enumerate(action_trace):
                    if step_limit is not None and step_i >= step_limit:
                        break

                    _obs, _reward, terminated, truncated, info = base_env.step(
                        int(action)
                    )
                    ended = bool(terminated or truncated)

                    # Track x during the replay.
                    try:
                        x_val = (
                            info.get("x") if isinstance(info, dict) else None
                        )
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
                            frame_bgr = cv2.cvtColor(
                                arr2, cv2.COLOR_RGB2BGR
                            )
                            text = (
                                f"x={int(last_x)}"
                                if last_x is not None
                                else "x=?"
                            )
                            writer.write(
                                _overlay_text(frame_bgr, text=text)
                            )

                    if ended:
                        break
            finally:
                writer.release()
        finally:
            base_env.close()

    # Best-effort rename.
    try:
        if tmp_video_path.exists():
            tmp_video_path.replace(out_path)
    except Exception:
        out_path = tmp_video_path

    if verbose:
        print(
            "[video-replay] wrote:",
            str(out_path),
            "recorded_max_x=",
            int(achieved_max_x) if achieved_max_x is not None else None,
        )

    return out_path, achieved_max_x


def _record_replay_video_worker(params: dict[str, Any], conn) -> None:
    # When the user hits Ctrl+C during training, the terminal sends SIGINT to
    # the whole foreground process group. We don't want the *video worker*
    # to print a KeyboardInterrupt traceback; the main training loop handles
    # the interrupt.
    ignore_sigint()

    try:
        max_steps_val = params.get("max_steps")
        max_steps_parsed: Optional[int]
        if max_steps_val is None:
            max_steps_parsed = None
        else:
            max_steps_parsed = int(max_steps_val)

        # Avoid passing non-picklable callables across processes.
        to_int = to_python_int
        out_path, achieved_max_x = _record_replay_video_impl(
            custom_data_root=str(params["custom_data_root"]),
            out_path=str(params["out_path"]),
            action_trace=list(params["action_trace"]),
            state_label=str(params["state_label"]),
            fps=int(params["fps"]),
            max_steps=max_steps_parsed,
            to_int=to_int,
            verbose=bool(params.get("verbose", False)),
        )
        conn.send(
            {
                "ok": True,
                "path": str(out_path),
                "achieved_max_x": achieved_max_x,
            }
        )
    except BaseException as e:
        # Includes KeyboardInterrupt/SystemExit. Best-effort error reporting
        # without emitting a traceback.
        try:
            conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def record_replay_video(
    *,
    custom_data_root: str,
    out_path: str | Path,
    action_trace: list[int],
    state_label: str,
    fps: int,
    max_steps: Optional[int] = None,
    to_int: Optional[Callable[[object], int]] = None,
    isolate_process: bool = True,
    timeout_s: int = 300,
    verbose: bool = False,
) -> tuple[Path, Optional[int]]:
    """Replay an action trace in a fresh env and record it to an MP4.

    Returns (final_path, achieved_max_x).
    """

    # Gym Retro only supports a single emulator instance per *process*.
    # Training already holds an emulator, so recording must happen in an
    # isolated subprocess.
    if isolate_process:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        params = {
            "custom_data_root": str(custom_data_root),
            "out_path": str(out_path),
            "action_trace": list(action_trace),
            "state_label": str(state_label),
            "fps": int(fps),
            "max_steps": max_steps,
            "verbose": bool(verbose),
        }
        proc = ctx.Process(
            target=_record_replay_video_worker,
            args=(params, child_conn),
            daemon=True,
        )
        proc.start()
        try:
            child_conn.close()
        except Exception:
            pass

        msg = None
        try:
            if parent_conn.poll(timeout=max(1, int(timeout_s))):
                msg = parent_conn.recv()
            else:
                raise TimeoutError(
                    f"Replay video worker timed out after {int(timeout_s)}s"
                )
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass
            proc.join(timeout=1)
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass

        if msg is None and proc.exitcode not in (None, 0):
            raise RuntimeError(
                f"Replay video worker exited with code {proc.exitcode}"
            )

        if not isinstance(msg, dict) or not msg.get("ok"):
            err = (
                "Unknown error"
                if not isinstance(msg, dict)
                else str(msg.get("error") or "Unknown error")
            )
            raise RuntimeError(err)

        return Path(str(msg["path"])), msg.get("achieved_max_x")

    to_int_fn = to_int if callable(to_int) else to_python_int
    return _record_replay_video_impl(
        custom_data_root=str(custom_data_root),
        out_path=str(out_path),
        action_trace=list(action_trace),
        state_label=str(state_label),
        fps=int(fps),
        max_steps=max_steps,
        to_int=to_int_fn,
        verbose=bool(verbose),
    )
