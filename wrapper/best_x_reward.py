from __future__ import annotations

from typing import Optional

import gymnasium as gym

from utils import to_python_int_or_none


class BestXRewardWrapper(gym.Wrapper):
    """Adds a bonus reward whenever the agent reaches a new best global X.

    Expects `info['x']` to be present (provided by DeathPositionLoggerWrapper).
    The best-x tracker resets every episode.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        bonus: float = 50.0,
        min_improvement_px: int = 10,
    ):
        super().__init__(env)
        self.bonus = float(bonus)
        self.min_improvement_px = max(1, int(min_improvement_px))
        self._best_x: Optional[int] = None
        self._next_bonus_x: Optional[int] = None

    def reset(self, **kwargs):
        self._best_x = None
        self._next_bonus_x = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info_out = dict(info)
        x_val = to_python_int_or_none(info_out.get("x"))

        bonus = 0.0
        if x_val is not None:
            if self._best_x is None:
                self._best_x = int(x_val)
                self._next_bonus_x = int(x_val) + int(self.min_improvement_px)
            else:
                if int(x_val) > int(self._best_x):
                    self._best_x = int(x_val)

                if (
                    self._next_bonus_x is not None
                    and int(x_val) >= int(self._next_bonus_x)
                ):
                    bonus = float(self.bonus)
                    self._next_bonus_x = int(x_val) + int(
                        self.min_improvement_px
                    )

        if bonus != 0.0:
            reward = float(reward) + bonus
            info_out["best_x_reward"] = bonus

        if self._best_x is not None:
            info_out["best_x"] = int(self._best_x)

        return obs, reward, terminated, truncated, info_out
