from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from tempfile import NamedTemporaryFile

from contextlib import contextmanager

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
        required_successes: int = 1,
        commit_switch: bool = True,
        emit_candidate_info: bool = True,
        next_state: Optional[str] = None,
        next_state_by_episode_state: Optional[Mapping[str, str]] = None,
        goal_x_by_episode_state: Optional[Mapping[str, int]] = None,
        shared_switch_path: Optional[str] = None,
    ):
        super().__init__(env)
        self.goal_x = int(goal_x)
        self.goal_reward = float(goal_reward)
        self.required_successes = max(1, int(required_successes))
        self.commit_switch = bool(commit_switch)
        self.emit_candidate_info = bool(emit_candidate_info)
        self.next_state = (
            _normalize_state_name(next_state)
            if next_state is not None
            else ""
        )
        self.next_state_by_episode_state = {
            _normalize_state_name(k): _normalize_state_name(v)
            for k, v in (next_state_by_episode_state or {}).items()
            if _normalize_state_name(k) and _normalize_state_name(v)
        }

        self.goal_x_by_episode_state = {
            _normalize_state_name(k): int(v)
            for k, v in (goal_x_by_episode_state or {}).items()
            if _normalize_state_name(k)
        }

        self._shared_switch_path = (
            Path(shared_switch_path)
            if shared_switch_path is not None
            and str(shared_switch_path).strip()
            else None
        )

        self._goal_reward_given: bool = False
        self._goal_success_counted: bool = False
        # If we apply the shared switch during reset, emit a one-time info flag
        # on the next step so callbacks can react.
        self._pending_switch_info: bool = False

    def _lock_path(self) -> Optional[Path]:
        if self._shared_switch_path is None:
            return None
        return Path(str(self._shared_switch_path) + ".lock")

    @contextmanager
    def _shared_lock(self):
        """Best-effort cross-process lock for the shared switch payload."""

        lock_path = self._lock_path()
        if lock_path is None:
            yield
            return

        try:
            import fcntl  # Unix-only

            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
            return
        except Exception:
            # If locking is not available, proceed without it.
            yield

    def _read_shared_payload(self) -> dict:
        if self._shared_switch_path is None:
            return {}
        try:
            if not self._shared_switch_path.exists():
                return {}
            contents = self._shared_switch_path.read_text(encoding="utf-8")
            data = json.loads(contents)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

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
        self._goal_success_counted = False
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
            data = self._read_shared_payload()
            state = data.get("next_state")
            if not isinstance(state, str):
                return None
            state = _normalize_state_name(state)
            return state if state else None
        except Exception:
            return None

    def _write_shared_payload(self, payload: dict) -> None:
        if self._shared_switch_path is None:
            return
        try:
            self._shared_switch_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _increment_success_and_maybe_switch(
        self, *, current_state: str, target_next: str
    ) -> tuple[int, bool]:
        """Increment success count for current_state and maybe commit switch.

        Returns: (success_count_for_current_state, did_switch_now)
        """

        if self._shared_switch_path is None:
            # No shared file configured: fall back to immediate switch.
            return self.required_successes, True

        with self._shared_lock():
            payload = self._read_shared_payload()

            existing_next = payload.get("next_state")
            if isinstance(existing_next, str) and _normalize_state_name(
                existing_next
            ):
                # Someone already committed a switch; don't override.
                return self.required_successes, False

            counts = payload.get("success_counts")
            if not isinstance(counts, dict):
                counts = {}

            cur_key = _normalize_state_name(current_state)
            prev = counts.get(cur_key, 0)
            try:
                prev_i = int(prev)
            except Exception:
                prev_i = 0
            new_count = prev_i + 1
            counts[cur_key] = int(new_count)
            payload["success_counts"] = counts

            did_switch = False
            if new_count >= self.required_successes:
                payload["next_state"] = _normalize_state_name(target_next)
                did_switch = True

            self._write_shared_payload(payload)
            return int(new_count), bool(did_switch)

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

        goal_x = self._choose_goal_x(info)
        if x_val is not None and x_val >= goal_x:
            info["goal_reached"] = True
            info["goal_x"] = int(goal_x)

            if not self._goal_reward_given:
                reward = float(reward) + float(self.goal_reward)
                self._goal_reward_given = True
                info["goal_reward"] = float(self.goal_reward)

            target_next = self._choose_next_state(info)

            # Always expose the candidate next state so an outer callback can
            # implement a deterministic eval gate.
            if self.emit_candidate_info and target_next:
                info["goal_candidate_next_state"] = _normalize_state_name(
                    target_next
                )

            # If switching is externally managed, don't count/commit here.
            if not self.commit_switch:
                return obs, reward, terminated, truncated, info

            if target_next and not self._goal_success_counted:
                self._goal_success_counted = True
                current_shared = self._read_shared_state()
                if current_shared:
                    # A shared switch is already committed; adopt it.
                    reset_wrapper = _find_wrapper(
                        self.env, ResetToDefaultStateByDeathWrapper
                    )
                    if reset_wrapper is not None:
                        reset_wrapper.default_state = current_shared
                        reset_wrapper._force_default_state_on_reset = True
                        info["level_switched"] = True
                        info["next_state"] = current_shared
                else:
                    raw_episode_state = info.get("episode_state")
                    cur_state = _normalize_state_name(str(raw_episode_state))
                    success_count, did_switch = (
                        self._increment_success_and_maybe_switch(
                            current_state=cur_state,
                            target_next=target_next,
                        )
                    )
                    info["goal_success_count"] = int(success_count)
                    info["goal_success_required"] = int(
                        self.required_successes
                    )

                    if did_switch:
                        reset_wrapper = _find_wrapper(
                            self.env, ResetToDefaultStateByDeathWrapper
                        )
                        if reset_wrapper is not None:
                            reset_wrapper.default_state = target_next
                            reset_wrapper._force_default_state_on_reset = True
                            info["level_switched"] = True
                            info["next_state"] = target_next

        return obs, reward, terminated, truncated, info

    def _choose_next_state(self, info: dict) -> str:
        """Pick the next state to switch to given current episode metadata."""

        raw_episode_state = info.get("episode_state")
        cur = _normalize_state_name(str(raw_episode_state))
        if cur and cur in self.next_state_by_episode_state:
            return self.next_state_by_episode_state[cur]
        return self.next_state

    def _choose_goal_x(self, info: dict) -> int:
        raw_episode_state = info.get("episode_state")
        cur = _normalize_state_name(str(raw_episode_state))
        if cur and cur in self.goal_x_by_episode_state:
            return int(self.goal_x_by_episode_state[cur])
        return int(self.goal_x)
