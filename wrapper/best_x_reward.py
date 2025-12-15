from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np


class BestXDeltaRewardWrapper(gym.Wrapper):
    """Reward shaping: reward only when reaching a new episode best X.

    Uses `info['x']` (global x, provided by DeathPositionLoggerWrapper).

    Behavior:
    - shaped_reward = max(0, x - best_x_so_far) * scale
    - optionally keep negative original rewards (penalties) by adding them
      back in. This keeps stillstand penalties while removing most positive
      game/score rewards.

    Notes:
    - Put this wrapper *inside* the GoalRewardAndStateSwitchWrapper so the
      goal bonus stays intact.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        scale: float = 1.0,
        keep_negative_original: bool = True,
    ):
        super().__init__(env)
        self.scale = float(scale)
        self.keep_negative_original = bool(keep_negative_original)
        self._best_x: Optional[int] = None

    @staticmethod
    def _to_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return int(np.asarray(value).item())

    def reset(self, **kwargs):
        self._best_x = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        x_val = self._to_int(info.get("x"))
        if x_val is None:
            return obs, reward, terminated, truncated, info

        if self._best_x is None:
            self._best_x = int(x_val)
            delta = 0
        else:
            delta = int(x_val) - int(self._best_x)
            if delta > 0:
                self._best_x = int(x_val)
            else:
                delta = 0

        shaped = float(delta) * float(self.scale)

        # Keep negative penalties (e.g. stuck penalty) if requested.
        if self.keep_negative_original and float(reward) < 0.0:
            shaped += float(reward)

        info["best_x"] = int(self._best_x)
        info["best_x_delta"] = int(delta)

        return obs, shaped, terminated, truncated, info
