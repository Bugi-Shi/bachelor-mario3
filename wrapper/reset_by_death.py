from typing import Any

import gymnasium as gym
import numpy as np
import retro.data as retro_data


class ResetToDefaultStateByDeathWrapper(gym.Wrapper):
    @staticmethod
    def _to_int(value) -> int:
        try:
            return int(value)
        except Exception:
            return int(np.asarray(value).item())

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
        self._prev_lives = None
        self._force_default_state_on_reset = True

    def reset(self, **kwargs):
        if self._force_default_state_on_reset:
            unwrapped: Any = self.env.unwrapped
            if hasattr(unwrapped, "load_state"):
                unwrapped.load_state(
                    self.default_state,
                    inttype=self.inttype,
                )
            self._force_default_state_on_reset = False
        self._prev_lives = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_lives = info.get(self.lives_key)
        if cur_lives is not None:
            cur_lives_int = self._to_int(cur_lives)

            if (
                self._prev_lives is not None
                and cur_lives_int < self._prev_lives
            ):
                self._force_default_state_on_reset = True
                terminated = True
                info = dict(info)
                info["life_lost_terminated"] = True
                info["life_lost_prev"] = self._prev_lives
                info["life_lost_cur"] = cur_lives_int

            self._prev_lives = cur_lives_int

        return obs, reward, terminated, truncated, info
