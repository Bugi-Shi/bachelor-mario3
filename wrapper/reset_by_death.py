from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import retro.data as retro_data

from utils import to_python_int


class ResetToDefaultStateByDeathWrapper(gym.Wrapper):
    """Resets the environment to a given save-state after losing a life.

    Detects life loss by watching `info[lives_key]` from the underlying env.
    When the value decreases compared to the previous step, the wrapper ends
    the episode (`terminated=True`) and forces the next `reset()` to load
    `default_state`.

    Adds two info fields on every `reset()` and `step()`:
    - `episode_state`: state name the current episode started from
    - `default_state`: configured default state
    """

    def __init__(
        self,
        env: gym.Env,
        default_state: str,
        inttype: retro_data.Integrations = retro_data.Integrations.CUSTOM_ONLY,
        lives_key: str = "lives",
    ):
        super().__init__(env)
        self.default_state = default_state
        self.inttype = inttype
        self.lives_key = lives_key
        self._prev_lives: Optional[int] = None
        self._force_default_state_on_reset: bool = True
        # Track which state the *current episode* started from.
        self._episode_state = default_state

    def reset(self, **kwargs):
        if self._force_default_state_on_reset:
            self._load_default_state_if_supported()
            self._episode_state = self.default_state
            self._force_default_state_on_reset = False

        self._prev_lives = None
        obs, info = self.env.reset(**kwargs)

        info_out = dict(info)
        info_out["episode_state"] = self._episode_state
        info_out["default_state"] = self.default_state
        return obs, info_out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info_out = dict(info)

        cur_lives = info_out.get(self.lives_key)
        if cur_lives is not None:
            cur_lives_int = to_python_int(cur_lives)

            if (
                self._prev_lives is not None
                and cur_lives_int < self._prev_lives
            ):
                self._force_default_state_on_reset = True
                terminated = True

                info_out["life_lost_terminated"] = True
                info_out["life_lost_prev"] = self._prev_lives
                info_out["life_lost_cur"] = cur_lives_int

            self._prev_lives = cur_lives_int

        info_out["episode_state"] = self._episode_state
        info_out["default_state"] = self.default_state
        return obs, reward, terminated, truncated, info_out

    def _load_default_state_if_supported(self) -> None:
        unwrapped: Any = self.env.unwrapped
        load_state = getattr(unwrapped, "load_state", None)
        if callable(load_state):
            load_state(self.default_state, inttype=self.inttype)
