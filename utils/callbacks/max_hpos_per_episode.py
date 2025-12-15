from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
                        float(r_val)
                        if r_val is not None
                        else float("nan")
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
                "custom/max_x_episode_mean",
                float(np.mean(ended_max_x)),
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
                "custom/episodes_ended_this_step",
                len(ended_envs),
            )
            # Make the metric people usually expect: cumulative episodes
            self.logger.record(
                "custom/episode_count",
                self._episode_count_total,
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
