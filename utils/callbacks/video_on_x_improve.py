from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gamebuilder.MB3_env import mariobros3_env


def _state_label(raw_state: Optional[object]) -> str:
    if raw_state is None:
        return "UnknownState"
    s = str(raw_state).strip()
    if not s:
        return "UnknownState"

    name = Path(s).name
    if name.endswith(".state"):
        name = name[: -len(".state")]

    if name.startswith("1Player."):
        name = name[len("1Player."):]

    # Filename-safe.
    name = name.replace("/", "_").replace("\\", "_")
    return name


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
            # Force the next reset to actually load the state.
            cur._force_default_state_on_reset = True
            return
        nxt = getattr(cur, "env", None)
        if nxt is None:
            return
        cur = nxt


class VideoOnXImproveCallback(BaseCallback):
    """Record a short rollout video whenever Mario reaches a new best X.

    Uses per-episode max global x from info['x'] (provided by
    DeathPositionLoggerWrapper during training). If info['x'] is not
    present, falls back to deriving global x from hpos + a wrap heuristic.

    Notes:
    - Designed for VecEnvs: episode boundaries detected via `dones`.
    - Records using a separate single-env evaluation env so training env isn't
      touched.
    """

    def __init__(
        self,
        *,
        custom_data_root: str,
        out_dir: str,
        n_stack: int = 4,
        min_improvement_x: int = 1,
        screen_width: int = 256,
        wrap_threshold: int = 50,
        wrap_prev_min: int = 200,
        wrap_cur_max: int = 60,
        wrap_cooldown_steps: int = 15,
        video_length_steps: int = 1500,
        fps: int = 30,
        deterministic: bool = False,
        min_episodes_before_trigger: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.custom_data_root = str(custom_data_root)
        self.out_dir = str(out_dir)
        self.n_stack = int(n_stack)
        self.min_improvement_x = int(min_improvement_x)

        # Match DeathPositionLoggerWrapper defaults so x aligns with
        # outputs/runs/*/deaths/deaths_env*.jsonl.
        self.screen_width = int(screen_width)
        self.wrap_threshold = int(wrap_threshold)
        self.wrap_prev_min = int(wrap_prev_min)
        self.wrap_cur_max = int(wrap_cur_max)
        self.wrap_cooldown_steps = int(wrap_cooldown_steps)

        self.video_length_steps = int(video_length_steps)
        self.fps = int(fps)
        self.deterministic = bool(deterministic)
        self.min_episodes_before_trigger = int(min_episodes_before_trigger)

        self._episodes_total = 0
        self._best_x: Optional[int] = None
        self._per_env_max_x: Optional[list[int]] = None
        self._printed_locals_keys: bool = False

        # Per-env tracking for screen wraps (same heuristic as logger wrapper)
        self._per_env_prev_hpos: Optional[list[Optional[int]]] = None
        self._per_env_screen_idx: Optional[list[int]] = None
        self._per_env_wrap_cooldown: Optional[list[int]] = None

    @staticmethod
    def _to_int(value) -> int:
        try:
            return int(value)
        except Exception:
            return int(np.asarray(value).item())

    def _init_callback(self) -> None:
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._per_env_max_x = [0 for _ in range(n_envs)]
        self._per_env_actions: list[list[int]] = [[] for _ in range(n_envs)]
        self._per_env_prev_hpos = [None for _ in range(n_envs)]
        self._per_env_screen_idx = [0 for _ in range(n_envs)]
        self._per_env_wrap_cooldown = [0 for _ in range(n_envs)]

    def _maybe_advance_screen(
        self, *, env_i: int, cur_hpos: int, ended: bool
    ) -> None:
        """Advance screen index using the same wrap heuristic as the logger."""

        if (
            ended
            or self._per_env_prev_hpos is None
            or self._per_env_screen_idx is None
            or self._per_env_wrap_cooldown is None
        ):
            return

        if self._per_env_wrap_cooldown[env_i] > 0:
            self._per_env_wrap_cooldown[env_i] -= 1

        prev = self._per_env_prev_hpos[env_i]
        if prev is None:
            self._per_env_prev_hpos[env_i] = int(cur_hpos)
            return

        # Heuristic: large drop without episode end -> likely screen boundary.
        if (
            self._per_env_wrap_cooldown[env_i] == 0
            and int(cur_hpos) + self.wrap_threshold < int(prev)
            and int(prev) >= self.wrap_prev_min
            and int(cur_hpos) <= self.wrap_cur_max
        ):
            self._per_env_screen_idx[env_i] += 1
            self._per_env_wrap_cooldown[env_i] = self.wrap_cooldown_steps

        self._per_env_prev_hpos[env_i] = int(cur_hpos)

    def _derived_global_x(
        self, *, env_i: int, info: dict, ended: bool
    ) -> Optional[int]:
        hpos = info.get("hpos")
        if hpos is None:
            return None

        # Prefer a real page/screen counter from RAM if available.
        world_x_hi = info.get("world_x_hi")
        if world_x_hi is not None:
            try:
                hi = self._to_int(world_x_hi)
            except Exception:
                hi = None
            if hi is not None:
                cur_hpos = self._to_int(hpos)
                return int(hi * self.screen_width + cur_hpos)

        if self._per_env_screen_idx is None:
            return None

        cur_hpos = self._to_int(hpos)
        self._maybe_advance_screen(env_i=env_i, cur_hpos=cur_hpos, ended=ended)
        return int(
            self._per_env_screen_idx[env_i] * self.screen_width + cur_hpos
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        actions = None
        for k in ("actions", "clipped_actions", "action", "clipped_action"):
            if k in self.locals:
                actions = self.locals.get(k)
                break
        if (
            infos is None
            or dones is None
            or self._per_env_max_x is None
            or self._per_env_prev_hpos is None
            or self._per_env_screen_idx is None
            or self._per_env_wrap_cooldown is None
        ):
            return True

        if self.verbose and not self._printed_locals_keys:
            try:
                print(
                    "[video-x-callback] locals keys:",
                    sorted(list(self.locals.keys())),
                )
            except Exception:
                pass
            self._printed_locals_keys = True

        ended_max_x: list[int] = []
        ended_envs: list[int] = []
        ended_action_traces: list[list[int]] = []
        ended_episode_states: list[str] = []

        saw_level_switch = False

        # Capture the actions actually taken during training so we can replay
        # the exact run that achieved a new best X.
        if actions is not None:
            try:
                # Support torch tensors without importing torch hard.
                if hasattr(actions, "detach") and hasattr(actions, "cpu"):
                    acts = actions.detach().cpu().numpy()  # type: ignore
                else:
                    acts = np.asarray(actions)

                # Expect per-env actions.
                if acts.ndim == 0:
                    acts = acts.reshape((1,))
                if acts.ndim == 2 and acts.shape[1] == 1:
                    acts = acts.reshape((acts.shape[0],))

                n_envs = len(infos)
                if acts.shape[0] >= n_envs:
                    for env_i in range(n_envs):
                        a_i = acts[env_i]
                        a_int = self._to_int(a_i)
                        self._per_env_actions[env_i].append(int(a_int))
            except Exception:
                pass

        for env_i, (done, info) in enumerate(zip(dones, infos)):
            ended = bool(done)
            if isinstance(info, dict):
                if info.get("level_switched"):
                    saw_level_switch = True

                # Prefer wrapper-provided info['x'].
                x_val = info.get("x")
                if x_val is not None:
                    cur_x = self._to_int(x_val)
                else:
                    # Fallback for setups without DeathPositionLoggerWrapper.
                    derived_x = self._derived_global_x(
                        env_i=env_i,
                        info=info,
                        ended=ended,
                    )
                    cur_x = int(derived_x) if derived_x is not None else None

                if cur_x is not None and cur_x > self._per_env_max_x[env_i]:
                    self._per_env_max_x[env_i] = int(cur_x)

            if ended:
                ended_max_x.append(int(self._per_env_max_x[env_i]))
                ended_envs.append(int(env_i))
                ended_action_traces.append(list(self._per_env_actions[env_i]))
                ended_episode_states.append(
                    _state_label(
                        (
                            info.get("episode_state")
                            if isinstance(info, dict)
                            else None
                        )
                    )
                )
                self._per_env_max_x[env_i] = 0
                self._per_env_actions[env_i].clear()

                # Reset per-env screen tracking for the next episode.
                self._per_env_prev_hpos[env_i] = None
                self._per_env_screen_idx[env_i] = 0
                self._per_env_wrap_cooldown[env_i] = 0

        # If we switched to the next level/state, reset the baseline so
        # improvements are evaluated fresh for the new state.
        # Do this even if no episode ended on this specific step.
        if saw_level_switch:
            self._best_x = None

        if not ended_max_x:
            return True

        self._episodes_total += len(ended_max_x)
        step_best_x = int(np.max(np.asarray(ended_max_x, dtype=np.int64)))

        self.logger.record("custom/episodes_total", self._episodes_total)
        self.logger.record(
            "custom/max_x_episode_mean",
            float(np.mean(ended_max_x)),
        )
        self.logger.record("custom/max_x_episode_best_this_step", step_best_x)

        if self._best_x is None:
            self._best_x = step_best_x
            self.logger.record("custom/x_video_baseline", float(self._best_x))
            return True

        # Avoid spamming early; let the policy stabilize first.
        if self._episodes_total < self.min_episodes_before_trigger:
            return True

        if step_best_x >= int(self._best_x) + max(1, self.min_improvement_x):
            try:
                # Record the *actual* episode by replaying its action trace.
                best_i = int(
                    np.argmax(np.asarray(ended_max_x, dtype=np.int64))
                )
                trace = ended_action_traces[best_i]
                state_label = (
                    ended_episode_states[best_i]
                    if best_i < len(ended_episode_states)
                    else "UnknownState"
                )
                trigger_ep = int(self._episodes_total)
                self._record_video(
                    step_best_x,
                    action_trace=trace,
                    state_label=state_label,
                    episode_num=trigger_ep,
                )
                self._best_x = step_best_x
                self.logger.record(
                    "custom/x_video_trigger_best_x",
                    float(step_best_x),
                )
            except Exception as e:
                if self.verbose:
                    print(
                        "[video-x-callback] Video recording failed: "
                        f"{type(e).__name__}: {e}"
                    )

        return True

    def _record_video(
        self,
        best_x: int,
        *,
        action_trace: list[int],
        state_label: str,
        episode_num: int,
    ) -> None:
        if self.model is None:
            return

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

        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename_tmp = (
            f"{state_label}-Ep_{int(episode_num):04d}_"
            f"bestx_{int(best_x):04d}.mp4"
        )
        out_path_tmp = out_dir / filename_tmp

        # Re-play the exact episode using the captured action trace.
        # Wrap with DeathPositionLoggerWrapper (logging to a temp file) so
        # info['x'] matches the training/death-logger logic.
        from wrapper.death_position_logger import DeathPositionLoggerWrapper

        tmp_deaths_dir = tempfile.mkdtemp(prefix="video_eval_deaths_")
        tmp_log_path = str(Path(tmp_deaths_dir) / "deaths_eval.jsonl")

        base_env = DeathPositionLoggerWrapper(
            mariobros3_env(
                self.custom_data_root,
                rank=0,
                run_dir=None,
                enable_death_logger=False,
                render_mode="rgb_array",
            ),
            log_path=tmp_log_path,
        )

        # Make sure the replay starts from the same state we name the video by.
        # Reverse the label mapping back to the common Retro state prefix.
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
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if callable(fourcc_fn):
            fourcc_val = fourcc_fn(*"mp4v")
            fourcc = int(fourcc_val)  # type: ignore[arg-type]
        else:
            fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))

        writer = cv2.VideoWriter(
            str(out_path_tmp),
            fourcc,
            float(self.fps),
            (width, height),
        )
        if not writer.isOpened():
            base_env.close()
            raise RuntimeError("Failed to open cv2.VideoWriter")

        achieved_max_x: Optional[int] = None
        last_x: Optional[int] = None
        death_x: Optional[int] = None

        try:
            first_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
            first_bgr = _overlay_text(first_bgr, text="x=?")
            writer.write(first_bgr)

            steps = len(action_trace)
            for i in range(steps):
                act = int(action_trace[i])
                _obs, _reward, terminated, truncated, info = base_env.step(act)
                ended = bool(terminated or truncated)

                # Track x during the replay.
                try:
                    x_val = info.get("x") if isinstance(info, dict) else None
                    if x_val is not None:
                        cur_x = self._to_int(x_val)
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
                            bgr2 = _overlay_text(bgr2, text=f"x={int(last_x)}")
                        else:
                            bgr2 = _overlay_text(bgr2, text="x=?")
                        writer.write(bgr2)

                if ended:
                    death_x = int(last_x) if last_x is not None else None
                    break
        finally:
            writer.release()
            base_env.close()

        # Rename to include the achieved max x, so filenames match what you
        # actually see in the video.
        if achieved_max_x is None:
            achieved_max_x = int(best_x)

        if death_x is None:
            if last_x is not None:
                death_x = int(last_x)
            else:
                death_x = int(achieved_max_x)

        # Requested filename scheme:
        # World1.LevelX-Ep_XXXX_bestx_XXXX.mp4
        # (episode number helps track spacing between videos).
        final_name = (
            f"{state_label}-Ep_{int(episode_num):04d}_"
            f"bestx_{int(best_x):04d}.mp4"
        )
        final_path = out_dir / final_name
        try:
            if out_path_tmp.exists():
                out_path_tmp.replace(final_path)
        except Exception:
            # If rename fails for any reason, keep the tmp filename.
            final_path = out_path_tmp

        if self.verbose:
            print(
                "[video-x-callback] wrote:",
                str(final_path),
                "trigger_x=",
                int(best_x),
                "recorded_max_x=",
                int(achieved_max_x),
            )
