from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import cv2
from stable_baselines3.common.callbacks import BaseCallback

from gamebuilder.MB3_env import mariobros3_env


class ResetStatsCallback(BaseCallback):
    def __init__(self, print_every_steps: int = 5000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_every_steps = int(print_every_steps)
        self.total_dones = 0
        self.stuck_dones = 0
        self.life_lost_dones = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is not None and dones is not None:
            for done, info in zip(dones, infos):
                if not done:
                    continue
                self.total_dones += 1
                if info.get("no_hpos_progress_terminated"):
                    self.stuck_dones += 1
                if info.get("life_lost_terminated"):
                    self.life_lost_dones += 1

        if self.print_every_steps and (
            self.n_calls % self.print_every_steps == 0
        ):
            print(
                "[reset-stats] dones=",
                self.total_dones,
                "stuck=",
                self.stuck_dones,
                "life_lost=",
                self.life_lost_dones,
            )
        return True


class MaxHposPerEpisodeCallback(BaseCallback):
    """Tracks max `info['hpos']` per episode and saves it.

    Notes:
    - Works with VecEnvs (SubprocVecEnv, DummyVecEnv, etc.).
    - Uses `dones` to detect episode boundaries.
    - Appends rows to a CSV so you can plot later.
    """

    def __init__(self, csv_path: str, window: int = 50, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.csv_path = str(csv_path)
        self.window = int(window)

        self._episode_idx = 0
        # Total episodes ended across all envs since training start.
        # Note: in VecEnvs typically only 1 env ends per step, so
        # "episodes ended this step" often stays at 1.
        self._episode_count_total = 0
        self._per_env_max_hpos: Optional[list[int]] = None
        self._per_env_max_x: Optional[list[int]] = None

        self._window_max_hpos: deque[int] = deque(maxlen=self.window)
        self._window_max_x: deque[int] = deque(maxlen=self.window)
        self._window_rewards: deque[float] = deque(maxlen=self.window)
        self._window_lens: deque[int] = deque(maxlen=self.window)

        self._header = (
            "episode,env,x,hpos,episode_reward,episode_len,life_lost,stuck\n"
        )

    @staticmethod
    def _to_int(value) -> int:
        try:
            return int(value)
        except Exception:
            return int(np.asarray(value).item())

    def _init_callback(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._per_env_max_hpos = [0 for _ in range(n_envs)]
        self._per_env_max_x = [0 for _ in range(n_envs)]

        csv_file = Path(self.csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        if csv_file.exists():
            try:
                first_line = csv_file.open("r", encoding="utf-8").readline()
            except Exception:
                first_line = ""
            if first_line and first_line != self._header:
                csv_file = csv_file.with_name(
                    f"{csv_file.stem}_v2{csv_file.suffix}"
                )
                self.csv_path = str(csv_file)
                if self.verbose:
                    print(
                        "[hpos-callback] Existing CSV schema differs; "
                        f"writing to: {self.csv_path}"
                    )

        csv_file.parent.mkdir(parents=True, exist_ok=True)
        if not csv_file.exists():
            csv_file.write_text(self._header, encoding="utf-8")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if (
            infos is None
            or dones is None
            or self._per_env_max_hpos is None
            or self._per_env_max_x is None
        ):
            return True

        ended_max_x: list[int] = []
        ended_max_hpos: list[int] = []
        ended_envs: list[int] = []
        ended_rewards: list[float] = []
        ended_lens: list[int] = []
        ended_life_lost: list[bool] = []
        ended_stuck: list[bool] = []

        for env_i, (done, info) in enumerate(zip(dones, infos)):
            hpos = info.get("hpos") if isinstance(info, dict) else None
            if hpos is not None:
                cur = self._to_int(hpos)
                if cur > self._per_env_max_hpos[env_i]:
                    self._per_env_max_hpos[env_i] = cur

            # Prefer the wrapper-provided global X if present.
            x_val = info.get("x") if isinstance(info, dict) else None
            # Fallback if the wrapper is not used but RAM fields exist.
            if x_val is None and isinstance(info, dict):
                world_x_hi = info.get("world_x_hi")
                if world_x_hi is not None and hpos is not None:
                    try:
                        x_val = (
                            self._to_int(world_x_hi) * 256
                            + self._to_int(hpos)
                        )
                    except Exception:
                        x_val = None
            if x_val is not None:
                cur_x = self._to_int(x_val)
                if cur_x > self._per_env_max_x[env_i]:
                    self._per_env_max_x[env_i] = cur_x

            if bool(done):
                max_x = int(self._per_env_max_x[env_i])
                max_hpos = int(self._per_env_max_hpos[env_i])
                ended_max_x.append(max_x)
                ended_max_hpos.append(max_hpos)
                ended_envs.append(env_i)

                ep = info.get("episode") if isinstance(info, dict) else None
                if isinstance(ep, dict):
                    r_val = ep.get("r")
                    ended_rewards.append(
                        float(r_val) if r_val is not None else float("nan")
                    )

                    l_val = ep.get("l")
                    ended_lens.append(int(l_val) if l_val is not None else -1)
                else:
                    ended_rewards.append(float("nan"))
                    ended_lens.append(-1)

                ended_life_lost.append(
                    bool(info.get("life_lost_terminated"))
                    if isinstance(info, dict)
                    else False
                )
                ended_stuck.append(
                    bool(info.get("no_hpos_progress_terminated"))
                    if isinstance(info, dict)
                    else False
                )

                self._per_env_max_hpos[env_i] = 0
                self._per_env_max_x[env_i] = 0

        if ended_envs:
            # Update totals + rolling window (per episode, not per step).
            self._episode_count_total += len(ended_envs)
            for i in range(len(ended_envs)):
                self._window_max_x.append(int(ended_max_x[i]))
                self._window_max_hpos.append(int(ended_max_hpos[i]))
                self._window_rewards.append(float(ended_rewards[i]))
                self._window_lens.append(int(ended_lens[i]))

            with Path(self.csv_path).open("a", encoding="utf-8") as f:
                for i, env_i in enumerate(ended_envs):
                    f.write(
                        f"{self._episode_idx},{env_i},"
                        f"{ended_max_x[i]},{ended_max_hpos[i]},"
                        f"{ended_rewards[i]},{ended_lens[i]},"
                        f"{int(ended_life_lost[i])},{int(ended_stuck[i])}\n"
                    )
                    self._episode_idx += 1

            # Per-step aggregates (only episodes that ended this env step)
            self.logger.record(
                "custom/max_x_episode_mean", float(np.mean(ended_max_x))
            )
            self.logger.record(
                "custom/max_hpos_episode_mean",
                float(np.mean(ended_max_hpos)),
            )
            self.logger.record(
                "custom/episode_reward_mean",
                float(np.nanmean(np.asarray(ended_rewards, dtype=np.float64))),
            )
            # Keep old meaning available explicitly
            self.logger.record(
                "custom/episodes_ended_this_step", len(ended_envs)
            )
            # Make the metric people usually expect: cumulative episodes
            self.logger.record(
                "custom/episode_count", self._episode_count_total
            )

            # Rolling window aggregates (across ended episodes)
            if self._window_max_x:
                self.logger.record(
                    "custom/max_x_episode_mean_window",
                    float(
                        np.mean(
                            np.asarray(
                                list(self._window_max_x),
                                dtype=np.float64,
                            )
                        )
                    ),
                )
            if self._window_max_hpos:
                self.logger.record(
                    "custom/max_hpos_episode_mean_window",
                    float(
                        np.mean(
                            np.asarray(
                                list(self._window_max_hpos),
                                dtype=np.float64,
                            )
                        )
                    ),
                )
            if self._window_rewards:
                self.logger.record(
                    "custom/episode_reward_mean_window",
                    float(
                        np.nanmean(
                            np.asarray(
                                list(self._window_rewards),
                                dtype=np.float64,
                            )
                        )
                    ),
                )
            if self._window_lens:
                self.logger.record(
                    "custom/episode_len_mean_window",
                    float(
                        np.mean(
                            np.asarray(
                                list(self._window_lens),
                                dtype=np.float64,
                            )
                        )
                    ),
                )

        return True


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

        # Capture the actions actually taken during training so we can replay
        # the exact run that achieved a new best X.
        if actions is not None:
            try:
                # Support torch tensors without importing torch hard.
                if hasattr(actions, "detach") and hasattr(actions, "cpu"):
                    acts = (
                        actions.detach().cpu().numpy()  # type: ignore
                    )
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
                # Prefer wrapper-provided info['x'].
                x_val = info.get("x")
                if x_val is not None:
                    cur_x = self._to_int(x_val)
                else:
                    # Fallback for setups without DeathPositionLoggerWrapper.
                    derived_x = self._derived_global_x(
                        env_i=env_i, info=info, ended=ended
                    )
                    cur_x = int(derived_x) if derived_x is not None else None

                if cur_x is not None and cur_x > self._per_env_max_x[env_i]:
                    self._per_env_max_x[env_i] = int(cur_x)

            if ended:
                ended_max_x.append(int(self._per_env_max_x[env_i]))
                ended_envs.append(int(env_i))
                ended_action_traces.append(list(self._per_env_actions[env_i]))
                self._per_env_max_x[env_i] = 0
                self._per_env_actions[env_i].clear()

                # Reset per-env screen tracking for the next episode.
                self._per_env_prev_hpos[env_i] = None
                self._per_env_screen_idx[env_i] = 0
                self._per_env_wrap_cooldown[env_i] = 0

        if not ended_max_x:
            return True

        self._episodes_total += len(ended_max_x)
        step_best_x = int(np.max(np.asarray(ended_max_x, dtype=np.int64)))

        self.logger.record("custom/episodes_total", self._episodes_total)
        self.logger.record(
            "custom/max_x_episode_mean", float(np.mean(ended_max_x))
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
                self._record_video(step_best_x, action_trace=trace)
                self._best_x = step_best_x
                self.logger.record(
                    "custom/x_video_trigger_best_x", float(step_best_x)
                )
            except Exception as e:
                if self.verbose:
                    print(
                        "[video-x-callback] Video recording failed: "
                        f"{type(e).__name__}: {e}"
                    )
        return True

    def _record_video(self, best_x: int, *, action_trace: list[int]) -> None:
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
            f"triggerx_{int(best_x)}_ep{self._episodes_total}_"
            f"t{self.num_timesteps}.mp4"
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
            str(out_path_tmp), fourcc, float(self.fps), (width, height)
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
                            bgr2 = _overlay_text(
                                bgr2,
                                text=f"x={int(last_x)}",
                            )
                        else:
                            bgr2 = _overlay_text(
                                bgr2,
                                text="x=?",
                            )
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

        final_name = (
            f"triggerx_{int(best_x)}_"
            f"recx_{int(achieved_max_x)}_"
            f"deathx_{int(death_x)}_"
            f"ep{self._episodes_total}_t{self.num_timesteps}.mp4"
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
