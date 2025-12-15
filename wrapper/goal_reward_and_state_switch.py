from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from wrapper.reset_by_death import ResetToDefaultStateByDeathWrapper


def _normalize_state_name(state: str) -> str:
    """Normalize user/state inputs to a Retro state name.

    Returns a state name without extension/path.
    """

    s = str(state).strip()
    if not s:
        return s

    # Allow passing full paths like
    # retro_custom/.../1Player.World1.Level3.state
    name = Path(s).name
    if name.endswith(".state"):
        name = name[: -len(".state")]

    # Some configs already omit extension (metadata.json does).
    return name


def _find_wrapper(env: gym.Env, wrapper_type: type) -> Optional[Any]:
    cur: Any = env
    while True:
        if isinstance(cur, wrapper_type):
            return cur
        nxt = getattr(cur, "env", None)
        if nxt is None:
            return None
        cur = nxt


class GoalRewardAndStateSwitchWrapper(gym.Wrapper):
    """Give a one-time goal reward and switch future resets to a new state.

        - If info['x'] reaches goal_x (inclusive), adds goal_reward once per
            episode.
        - On first goal reach, updates
            ResetToDefaultStateByDeathWrapper.default_state
      to next_state and forces that state to be loaded on the next reset.

    Assumes DeathPositionLoggerWrapper (or similar) provides info['x'].
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        goal_x: int,
        goal_reward: float,
        next_state: str,
    ):
        super().__init__(env)
        self.goal_x = int(goal_x)
        self.goal_reward = float(goal_reward)
        self.next_state = _normalize_state_name(next_state)

        self._goal_reward_given: bool = False
        self._switched: bool = False

    @staticmethod
    def _to_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return int(np.asarray(value).item())

    def reset(self, **kwargs):
        self._goal_reward_given = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        x_val = self._to_int(info.get("x"))

        if x_val is not None and x_val >= self.goal_x:
            info["goal_reached"] = True
            info["goal_x"] = int(self.goal_x)

            if not self._goal_reward_given:
                reward = float(reward) + float(self.goal_reward)
                self._goal_reward_given = True
                info["goal_reward"] = float(self.goal_reward)

            if not self._switched and self.next_state:
                reset_wrapper = _find_wrapper(
                    self.env, ResetToDefaultStateByDeathWrapper
                )
                if reset_wrapper is not None:
                    reset_wrapper.default_state = self.next_state
                    # Ensure the next episode actually starts from it.
                    reset_wrapper._force_default_state_on_reset = True
                    info["level_switched"] = True
                    info["next_state"] = self.next_state
                self._switched = True

        return obs, reward, terminated, truncated, info
