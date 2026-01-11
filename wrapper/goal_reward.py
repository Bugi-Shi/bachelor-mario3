from __future__ import annotations
import gymnasium as gym
from utils import to_python_int_or_none


class GoalRewardWrapper(gym.Wrapper):
    """Give a one-time goal reward once `info['x']` reaches a threshold.

    Behavior:
    - Emits `info['goal_reached']` once `x >= goal_x`.
    - Adds `goal_reward` once per episode when the goal is first reached.

    For compatibility with GoalWindowGateCallback, it can optionally emit a
    stable `info['goal_candidate_next_state']` string when the goal is reached.

    Assumes DeathPositionLoggerWrapper (or similar) provides `info['x']`.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        goal_x: int,
        goal_reward: float,
        emit_candidate_info: bool = True,
        candidate_next_state: str = "LevelCleared",
    ):
        super().__init__(env)
        self.goal_x = int(goal_x)
        self.goal_reward = float(goal_reward)
        self.emit_candidate_info = bool(emit_candidate_info)
        self.candidate_next_state = str(candidate_next_state).strip()

        self._goal_reward_given: bool = False

    def reset(self, **kwargs):
        self._goal_reward_given = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        x_val = to_python_int_or_none(info.get("x"))

        if x_val is not None and x_val >= int(self.goal_x):
            info["goal_reached"] = True
            info["goal_x"] = int(self.goal_x)

            if not self._goal_reward_given:
                reward = float(reward) + float(self.goal_reward)
                self._goal_reward_given = True
                info["goal_reward"] = float(self.goal_reward)

            if self.emit_candidate_info:
                info["goal_candidate_next_state"] = (
                    self.candidate_next_state
                    or str(info.get("episode_state") or "")
                )

        return obs, reward, terminated, truncated, info
