from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


def _normalize_state_name(raw: object) -> str:
    s = str(raw).strip()
    if not s:
        return ""
    name = Path(s).name
    if name.endswith(".state"):
        name = name[: -len(".state")]
    return name


class GoalWindowGateCallback(BaseCallback):
    """Gate a level switch based on repeated goal reaches within a window.

    Desired behavior:
    - When an episode reaches the goal (info['goal_reached'] at least once), we
      count it as a success event for the current (start_state -> next_state)
      pair.
        - If we observe `required_successes` goal-success episodes total
            (not necessarily consecutive; global across all envs), we either:
                - stop training (if stop_training=True), or
                - commit the switch by writing `level_switch.json`.

    This replaces the previous deterministic eval-env gate.
    """

    def __init__(
        self,
        *,
        shared_switch_path: str,
        target_start_state: str = "",
        required_successes: int = 3,
        window_episodes: int = 10,
        stop_training: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.shared_switch_path = str(shared_switch_path)
        self.target_start_state = _normalize_state_name(target_start_state)
        self.required_successes = max(1, int(required_successes))
        # Kept for backwards compatibility; no longer affects gating.
        self.window_episodes = max(1, int(window_episodes))
        self.stop_training = bool(stop_training)

        self._episode_idx: int = 0
        self._per_env_goal_reached: Optional[list[bool]] = None
        self._per_env_start_state: Optional[list[str]] = None
        self._per_env_next_state: Optional[list[str]] = None

        # Active gate key: successes count only for this start->next pair.
        self._gate_start_state: str = ""
        self._gate_next_state: str = ""

        # Count of goal-success episodes for the active start->next pair.
        self._success_count: int = 0

    def _init_callback(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._per_env_goal_reached = [False for _ in range(n_envs)]
        self._per_env_start_state = ["" for _ in range(n_envs)]
        self._per_env_next_state = ["" for _ in range(n_envs)]

    def _read_committed_next_state(self) -> str:
        p = Path(self.shared_switch_path)
        try:
            if not p.exists():
                return ""
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return ""
            val = data.get("next_state")
            return str(val).strip() if isinstance(val, str) else ""
        except Exception:
            return ""

    def _write_next_state(self, *, next_state: str, meta: dict) -> None:
        p = Path(self.shared_switch_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"next_state": str(next_state), **meta}
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(p.parent),
            prefix=p.name + ".tmp.",
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(json.dumps(payload, ensure_ascii=False))
        tmp_path.replace(p)

    def _reset_gate(self) -> None:
        self._gate_start_state = ""
        self._gate_next_state = ""
        self._success_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if (
            infos is None
            or dones is None
            or self._per_env_goal_reached is None
            or self._per_env_start_state is None
            or self._per_env_next_state is None
        ):
            return True

        # If a switch is already committed, do nothing.
        # (Wrappers will adopt it.)
        if (not self.stop_training) and self._read_committed_next_state():
            return True

        # Track goal reached within an episode.
        for env_i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            # Reset internal gate state when an actual switch gets applied.
            # (The env wrapper emits this once after adopting shared state.)
            if info.get("level_switched"):
                self._reset_gate()

            if info.get("goal_reached"):
                self._per_env_goal_reached[env_i] = True
                self._per_env_start_state[env_i] = _normalize_state_name(
                    info.get("episode_state")
                )
                self._per_env_next_state[env_i] = _normalize_state_name(
                    info.get("goal_candidate_next_state")
                )

        # On episode end, count success events.
        for env_i, (done, info) in enumerate(zip(dones, infos)):
            if not bool(done):
                continue

            cur_ep = int(self._episode_idx)
            self._episode_idx += 1

            if not bool(self._per_env_goal_reached[env_i]):
                # Reset per-episode flags.
                self._per_env_goal_reached[env_i] = False
                self._per_env_start_state[env_i] = ""
                self._per_env_next_state[env_i] = ""
                continue

            start_state = str(self._per_env_start_state[env_i] or "")
            next_state = str(self._per_env_next_state[env_i] or "")

            # Reset per-episode flags.
            self._per_env_goal_reached[env_i] = False
            self._per_env_start_state[env_i] = ""
            self._per_env_next_state[env_i] = ""

            if not start_state or not next_state:
                continue

            # Optional filter: only count successes from a specific
            # start state.
            if self.target_start_state and (
                start_state != self.target_start_state
            ):
                continue

            # First success defines the gate target pair.
            if not self._gate_start_state or not self._gate_next_state:
                self._gate_start_state = start_state
                self._gate_next_state = next_state

            # Only count successes for the currently active pair.
            if (
                start_state != self._gate_start_state
                or next_state != self._gate_next_state
            ):
                continue

            self._success_count += 1

            if self.verbose:
                print(
                    "[goal-window-gate] success",
                    f"ep={cur_ep}",
                    "count=",
                    int(self._success_count),
                    "needed=",
                    int(self.required_successes),
                    "from=",
                    self._gate_start_state,
                    "to=",
                    self._gate_next_state,
                )

            if int(self._success_count) >= int(self.required_successes):
                if self.stop_training:
                    if self.verbose:
                        print(
                            "[goal-window-gate] stopping training:",
                            (
                                f"{self.required_successes} total successes"
                            ),
                            "from=",
                            self._gate_start_state,
                            "to=",
                            self._gate_next_state,
                        )
                    return False

                meta = {
                    "source": "GoalWindowGateCallback",
                    "from_state": self._gate_start_state,
                    "required_successes": int(self.required_successes),
                    "episode_idx": int(cur_ep),
                }
                self._write_next_state(
                    next_state=self._gate_next_state,
                    meta=meta,
                )
                if self.verbose:
                    print(
                        "[goal-window-gate] committed next_state:",
                        self._gate_next_state,
                    )
                # After committing, stop counting.
                self._reset_gate()
                return True

        return True
