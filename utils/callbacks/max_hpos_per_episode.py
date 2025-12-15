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

        self._last_episode_state: Optional[str] = None

        self._last_level_code: Optional[str] = None

        self._header = (
            "episode,env,level,start_state,x,hpos,episode_reward,episode_len,"
            "life_lost,stuck\n"
        )

    @staticmethod
    def _normalize_state_name(raw_state: object) -> str:
        s = str(raw_state).strip()
        if not s:
            return ""
        name = Path(s).name
        if name.endswith(".state"):
            name = name[: -len(".state")]
        return name

    @classmethod
    def _level_code(cls, raw_state: object) -> str:
        """Return a short level code like '1-1' / '1-2' / '1-3' / '1-6'."""

        s = cls._normalize_state_name(raw_state)
        if not s:
            return "Unknown"

        # Strip optional prefix.
        if s.startswith("1Player."):
            s = s[len("1Player."):]

        import re

        m_world = re.search(r"World(?P<w>\d+)", s)
        world = m_world.group("w") if m_world else "1"

        if re.search(r"Airship", s, flags=re.IGNORECASE):
            return f"{world}-A"
        if re.search(r"MiniFortress", s, flags=re.IGNORECASE):
            return f"{world}-MF"

        # Special curriculum state: treat as the same level (1-2).
        if re.search(r"Level2_Pit", s, flags=re.IGNORECASE):
            return f"{world}-2"

        m_level = re.search(r"Level(?P<l>\d+)", s)
        if m_level:
            return f"{world}-{m_level.group('l')}"

        return "Unknown"

    @classmethod
    def _level_id(cls, raw_state: object) -> int:
        code = cls._level_code(raw_state)
        mapping = {
            "1-1": 11,
            "1-2": 12,
            "1-3": 13,
            "1-6": 16,
            "1-A": 110,
            "1-MF": 111,
        }
        return int(mapping.get(code, 0))

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

        def _next_versioned_path(p: Path) -> Path:
            """Return p or a _vN variant that matches the current header."""

            cur = p
            while cur.exists():
                try:
                    first_line = cur.open("r", encoding="utf-8").readline()
                except Exception:
                    first_line = ""
                if first_line == self._header:
                    return cur

                stem = cur.stem
                import re

                m = re.match(r"^(?P<base>.*)_v(?P<n>\d+)$", stem)
                if m:
                    base = m.group("base")
                    n = int(m.group("n")) + 1
                else:
                    base = stem
                    n = 2

                cur = cur.with_name(f"{base}_v{n}{cur.suffix}")

            return cur

        csv_file = _next_versioned_path(csv_file)
        self.csv_path = str(csv_file)
        if self.verbose and Path(self.csv_path) != Path(csv_file):
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
        ended_states: list[str] = []
        ended_levels: list[str] = []

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

                episode_state = (
                    self._normalize_state_name(info.get("episode_state"))
                    if isinstance(info, dict)
                    else ""
                )
                ended_states.append(episode_state)
                ended_levels.append(
                    self._level_code(episode_state) if episode_state else ""
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
                        f"{ended_levels[i] if i < len(ended_levels) else ''},"
                        f"{ended_states[i] if i < len(ended_states) else ''},"
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

            # Help debugging when switching levels: record a numeric level id.
            if ended_states:
                last_state = ended_states[-1]
                if last_state and last_state != (
                    self._last_episode_state or ""
                ):
                    self._last_episode_state = last_state

                last_level = self._level_code(last_state) if last_state else ""
                if last_level and last_level != (self._last_level_code or ""):
                    self._last_level_code = last_level
                    if self.verbose:
                        print("[hpos-callback] level:", last_level)

                # Human-readable in the SB3 terminal table.
                if last_level:
                    self.logger.record(
                        f"custom/current_level_{last_level}",
                        1.0,
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
