from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .video_replay import record_replay_video


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


def _short_level_label(state_label: str) -> str:
    """Return a short label used for filenames.

    Examples:
        "World1.Level1" -> "Level1"
        "World1.Level2_Pit" -> "Level2_Pit"
        "World1.Airship" -> "Airship"
    """

    s = str(state_label).strip()
    if not s or s == "UnknownState":
        return "Unknown"

    # Keep only the suffix after the last dot (drops "WorldX.").
    if "." in s:
        s = s.split(".")[-1]

    # Filename-safe.
    s = s.replace("/", "_").replace("\\", "_")
    return s or "Unknown"


class VideoOnXImproveCallback(BaseCallback):
    """Record best-x videos until first goal, then goal-only.

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
        fallback_goal_x: Optional[int] = None,
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
        self.fallback_goal_x = (
            int(fallback_goal_x) if fallback_goal_x is not None else None
        )

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
        self._goal_only: bool = False
        self._per_env_max_x: Optional[list[int]] = None
        self._per_env_goal_x: Optional[list[Optional[int]]] = None
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
        self._per_env_goal_reached: list[bool] = [False for _ in range(n_envs)]
        self._per_env_goal_x = [
            None for _ in range(n_envs)
        ]
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
            or self._per_env_goal_x is None
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
        ended_goal_reached: list[bool] = []
        ended_goal_x: list[Optional[int]] = []

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

                # Track whether the current episode reached the goal at least
                # once.
                # GoalRewardAndStateSwitchWrapper sets info['goal_reached']
                # when x>=goal_x.
                try:
                    if info.get("goal_reached"):
                        self._goal_only = True
                        self._per_env_goal_reached[env_i] = True
                        gx = info.get("goal_x")
                        if gx is not None:
                            try:
                                self._per_env_goal_x[env_i] = self._to_int(gx)
                            except Exception:
                                pass
                except Exception:
                    pass

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
                ended_goal_reached.append(
                    bool(self._per_env_goal_reached[env_i])
                )
                ended_goal_x.append(self._per_env_goal_x[env_i])
                self._per_env_max_x[env_i] = 0
                self._per_env_actions[env_i].clear()
                self._per_env_goal_reached[env_i] = False
                self._per_env_goal_x[env_i] = None

                # Reset per-env screen tracking for the next episode.
                self._per_env_prev_hpos[env_i] = None
                self._per_env_screen_idx[env_i] = 0
                self._per_env_wrap_cooldown[env_i] = 0

        # If we switched to the next level/state, reset the baseline so
        # improvements are evaluated fresh for the new state.
        # Do this even if no episode ended on this specific step.
        if saw_level_switch:
            self._best_x = None
            self._goal_only = False

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

        # Only record videos for episodes that actually reached the goal, and
        # only when we can confirm max_x>=goal_x.
        goal_candidate_idxs: list[int] = []
        for i, gr in enumerate(ended_goal_reached):
            if not bool(gr):
                continue
            gx = ended_goal_x[i]
            if gx is None:
                gx = self.fallback_goal_x
            if gx is None:
                continue
            if int(ended_max_x[i]) >= int(gx):
                goal_candidate_idxs.append(int(i))

        if goal_candidate_idxs:
            # Once the goal is reached at least once for the current level,
            # switch to goal-only videos until a state switch resets us.
            self._goal_only = True

            # Record one goal video per callback step: pick the goal-reaching
            # ended episode with the largest max_x.
            best_i = max(
                goal_candidate_idxs,
                key=lambda i: int(ended_max_x[i]),
            )
            candidate_x = int(ended_max_x[best_i])
            try:
                trace = ended_action_traces[best_i]
                state_label = (
                    ended_episode_states[best_i]
                    if best_i < len(ended_episode_states)
                    else "UnknownState"
                )
                short_level = _short_level_label(state_label)
                trigger_ep = int(self._episodes_total)
                self._record_video(
                    candidate_x,
                    action_trace=trace,
                    state_label=state_label,
                    episode_num=trigger_ep,
                    level_name=short_level,
                )
                self.logger.record(
                    "custom/x_video_trigger_best_x",
                    float(candidate_x),
                )
            except Exception as e:
                if self.verbose:
                    print(
                        "[video-x-callback] Goal video recording failed: "
                        f"{type(e).__name__}: {e}"
                    )
            return True

        # No goal-qualified episode ended on this step.
        if self._goal_only:
            return True

        # Pre-goal phase: record best-x improvement videos.
        if self._best_x is None:
            self._best_x = step_best_x
            self.logger.record("custom/x_video_baseline", float(self._best_x))
            return True

        self.logger.record("custom/x_video_baseline", float(self._best_x))

        # Avoid spamming early; let the policy stabilize first.
        if self._episodes_total < self.min_episodes_before_trigger:
            return True

        if step_best_x >= int(self._best_x) + max(1, self.min_improvement_x):
            try:
                best_i = int(
                    np.argmax(np.asarray(ended_max_x, dtype=np.int64))
                )
                trace = ended_action_traces[best_i]
                state_label = (
                    ended_episode_states[best_i]
                    if best_i < len(ended_episode_states)
                    else "UnknownState"
                )
                short_level = _short_level_label(state_label)
                trigger_ep = int(self._episodes_total)
                self._record_video(
                    step_best_x,
                    action_trace=trace,
                    state_label=state_label,
                    episode_num=trigger_ep,
                    level_name=short_level,
                )
                self._best_x = int(step_best_x)
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
        level_name: str,
    ) -> None:
        if self.model is None:
            return

        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        level_name = _short_level_label(level_name)

        final_name = (
            f"{level_name}_Ep_{int(episode_num):04d}_"
            f"bestx_{int(best_x):04d}.mp4"
        )
        final_path = out_dir / final_name

        written_path, achieved_max_x = record_replay_video(
            custom_data_root=self.custom_data_root,
            out_path=final_path,
            action_trace=action_trace,
            state_label=state_label,
            fps=int(self.fps),
            max_steps=int(self.video_length_steps),
            isolate_process=True,
            verbose=bool(self.verbose),
        )

        if self.verbose:
            print(
                "[video-x-callback] wrote:",
                str(written_path),
                "trigger_x=",
                int(best_x),
                "recorded_max_x=",
                int(achieved_max_x) if achieved_max_x is not None else None,
            )
