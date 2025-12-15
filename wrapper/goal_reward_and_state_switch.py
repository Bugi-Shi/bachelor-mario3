from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from tempfile import NamedTemporaryFile

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
        shared_switch_path: Optional[str] = None,
    ):
        super().__init__(env)
        self.goal_x = int(goal_x)
        self.goal_reward = float(goal_reward)
        self.next_state = _normalize_state_name(next_state)

        self._shared_switch_path = (
            Path(shared_switch_path)
            if shared_switch_path is not None
            and str(shared_switch_path).strip()
            else None
        )

        self._goal_reward_given: bool = False
        self._switched: bool = False
        # If we apply the shared switch during reset, emit a one-time info flag
        # on the next step so callbacks can react.
        self._pending_switch_info: bool = False

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
        self._maybe_apply_shared_switch_before_reset()
        return self.env.reset(**kwargs)

    def _maybe_apply_shared_switch_before_reset(self) -> None:
        """If a shared switch flag exists, ensure the next reset loads it."""

        if self._shared_switch_path is None:
            return

        state = self._read_shared_state()
        if not state:
            return

        reset_wrapper = _find_wrapper(
            self.env, ResetToDefaultStateByDeathWrapper
        )
        if reset_wrapper is None:
            return

        # Only mark as pending if this env hasn't adopted it yet.
        if getattr(reset_wrapper, "default_state", None) != state:
            reset_wrapper.default_state = state
            reset_wrapper._force_default_state_on_reset = True
            self._pending_switch_info = True

    def _read_shared_state(self) -> Optional[str]:
        if self._shared_switch_path is None:
            return None
        try:
            if not self._shared_switch_path.exists():
                return None
            contents = self._shared_switch_path.read_text(encoding="utf-8")
            data = json.loads(contents)
            if not isinstance(data, dict):
                return None
            state = data.get("next_state")
            if not isinstance(state, str):
                return None
            state = _normalize_state_name(state)
            return state if state else None
        except Exception:
            return None

    def _write_shared_state(self, *, state: str) -> None:
        if self._shared_switch_path is None:
            return
        try:
            self._shared_switch_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"next_state": _normalize_state_name(state)}
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(self._shared_switch_path.parent),
                prefix=self._shared_switch_path.name + ".tmp.",
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(json.dumps(payload, ensure_ascii=False))
            tmp_path.replace(self._shared_switch_path)
        except Exception:
            # Best-effort: training should not crash if writing fails.
            return

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        x_val = self._to_int(info.get("x"))

        if self._pending_switch_info:
            self._pending_switch_info = False
            info["level_switched"] = True
            shared_state = self._read_shared_state()
            if shared_state:
                info["next_state"] = shared_state

        if x_val is not None and x_val >= self.goal_x:
            info["goal_reached"] = True
            info["goal_x"] = int(self.goal_x)

            if not self._goal_reward_given:
                reward = float(reward) + float(self.goal_reward)
                self._goal_reward_given = True
                info["goal_reward"] = float(self.goal_reward)

            if not self._switched and self.next_state:
                # Persist the switch so all env processes can pick it up.
                if self._shared_switch_path is not None:
                    self._write_shared_state(state=self.next_state)

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
