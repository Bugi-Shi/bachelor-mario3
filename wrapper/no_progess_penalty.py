import gymnasium as gym
import numpy as np


class NoHposProgressGuardWrapper(gym.Wrapper):
    """Adds a penalty when Mario is stuck and terminates after N stuck steps.

    This merges the old "penalty" and "terminate" logic so you only have one
    place to tune the behavior.
    """

    def __init__(
        self,
        env,
        penalty: float = 0.2,
        max_no_progress_steps: int = 240,
        min_progress: int = 2,
        jitter_tolerance: int = 1,
    ):
        super().__init__(env)
        if max_no_progress_steps <= 0:
            raise ValueError("max_no_progress_steps must be > 0")
        if min_progress <= 0:
            raise ValueError("min_progress must be > 0")
        if jitter_tolerance < 0:
            raise ValueError("jitter_tolerance must be >= 0")

        self.penalty = float(penalty)
        self.max_no_progress_steps = int(max_no_progress_steps)
        self.min_progress = int(min_progress)
        self.jitter_tolerance = int(jitter_tolerance)

        self._prev_hpos = None
        self._no_progress_steps = 0

    def reset(self, **kwargs):
        self._prev_hpos = None
        self._no_progress_steps = 0
        return self.env.reset(**kwargs)

    @staticmethod
    def _to_int(value) -> int:
        try:
            return int(value)
        except Exception:
            return int(np.asarray(value).item())

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cur = info.get("hpos")
        if cur is None:
            return obs, reward, terminated, truncated, info

        cur_hpos = self._to_int(cur)

        if self._prev_hpos is None:
            self._prev_hpos = cur_hpos
            self._no_progress_steps = 0
            return obs, reward, terminated, truncated, info

        delta = cur_hpos - self._prev_hpos
        self._prev_hpos = cur_hpos

        # If we make real forward progress, clear the stuck counter.
        if delta >= self.min_progress:
            self._no_progress_steps = 0
            return obs, reward, terminated, truncated, info

        # "Stuck" means hpos is basically not changing
        # (pipe jitter, hanging, etc.).
        if abs(delta) <= self.jitter_tolerance:
            reward = float(reward) - self.penalty
            self._no_progress_steps += 1
            if self._no_progress_steps >= self.max_no_progress_steps:
                terminated = True
                info = dict(info)
                info["no_hpos_progress_terminated"] = True
                info["no_hpos_progress_steps"] = self._no_progress_steps
                info["no_hpos_progress_delta"] = delta
                info["no_hpos_progress_min_progress"] = self.min_progress
                info["no_hpos_progress_jitter_tol"] = self.jitter_tolerance
            return obs, reward, terminated, truncated, info

        # If we are moving (even left), don't count as stuck.
        self._no_progress_steps = 0
        return obs, reward, terminated, truncated, info
