from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gamebuilder.MB3_env import mariobros3_env


def _normalize_state_name(state: object) -> str:
    s = str(state).strip()
    if not s:
        return ""
    name = Path(s).name
    if name.endswith(".state"):
        name = name[: -len(".state")]
    return name


def _set_env_default_state(env: Any, *, state: str) -> None:
    """Set ResetToDefaultStateByDeathWrapper.default_state inside env."""

    try:
        from wrapper.reset_by_death import ResetToDefaultStateByDeathWrapper
    except Exception:
        return

    cur: Any = env
    while True:
        if isinstance(cur, ResetToDefaultStateByDeathWrapper):
            cur.default_state = str(state)
            cur._force_default_state_on_reset = True
            return
        nxt = getattr(cur, "env", None)
        if nxt is None:
            return
        cur = nxt


class LevelGateEvalCallback(BaseCallback):
    """Deterministic eval gate for level switching.

    Idea:
    - Training envs can report `info['goal_reached']` and a candidate
      `info['goal_candidate_next_state']`.
    - When that happens, run N evaluation rollouts (single env, deterministic)
      from the same start state.
    - Only if all N succeed, write `level_switch.json` (shared_switch_path)
      so training envs adopt the next state on their next reset.

    This avoids the "one of many parallel envs got lucky" problem.
    """

    def __init__(
        self,
        *,
        custom_data_root: str,
        shared_switch_path: str,
        required_successes: int = 3,
        eval_max_steps: int = 6000,
        deterministic: bool = True,
        cooldown_steps: int = 50_000,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.custom_data_root = str(custom_data_root)
        self.shared_switch_path = str(shared_switch_path)
        self.required_successes = max(1, int(required_successes))
        self.eval_max_steps = max(1, int(eval_max_steps))
        self.deterministic = bool(deterministic)
        self.cooldown_steps = max(0, int(cooldown_steps))

        self._eval_env = None
        self._last_eval_trigger_step: int = -10**18

    def _init_callback(self) -> None:
        # A plain gym env (not VecEnv) is fine for model.predict().
        self._eval_env = mariobros3_env(
            self.custom_data_root,
            rank=0,
            run_dir=None,
            enable_death_logger=False,
            render_mode=None,
        )

    def _on_training_end(self) -> None:
        try:
            if self._eval_env is not None:
                self._eval_env.close()
        except Exception:
            pass
        self._eval_env = None

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

    @staticmethod
    def _action_to_int(action) -> int:
        try:
            arr = np.asarray(action)
            if arr.ndim == 0:
                return int(arr.item())
            return int(arr.reshape((-1,))[0])
        except Exception:
            return int(action)

    def _eval_once(self, *, start_state: str) -> bool:
        if self._eval_env is None or self.model is None:
            return False

        _set_env_default_state(self._eval_env, state=start_state)
        obs, _info = self._eval_env.reset()

        for _ in range(self.eval_max_steps):
            act, _ = self.model.predict(obs, deterministic=self.deterministic)
            act_i = self._action_to_int(act)
            obs, _rew, terminated, truncated, info = self._eval_env.step(act_i)

            if isinstance(info, dict) and info.get("goal_reached"):
                return True

            if bool(terminated or truncated):
                return False

        return False

    def _run_eval_gate(self, *, start_state: str) -> bool:
        successes = 0
        for _ in range(self.required_successes):
            ok = self._eval_once(start_state=start_state)
            if not ok:
                return False
            successes += 1
        return successes >= self.required_successes

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None or self.model is None:
            return True

        # Cooldown to prevent repeated eval spam.
        if self.cooldown_steps > 0:
            since = int(self.num_timesteps) - int(self._last_eval_trigger_step)
            if int(since) < int(self.cooldown_steps):
                return True

        # Only act if nothing is committed yet.
        if self._read_committed_next_state():
            return True

        trigger_info: Optional[dict] = None
        for info in infos:
            if not isinstance(info, dict):
                continue
            if not info.get("goal_reached"):
                continue
            if not info.get("goal_candidate_next_state"):
                continue
            trigger_info = info
            break

        if trigger_info is None:
            return True

        start_state = _normalize_state_name(trigger_info.get("episode_state"))
        next_state = _normalize_state_name(
            trigger_info.get("goal_candidate_next_state")
        )
        if not start_state or not next_state:
            return True

        self._last_eval_trigger_step = int(self.num_timesteps)

        if self.verbose:
            print(
                "[level-gate] trigger: start_state=",
                start_state,
                "candidate_next=",
                next_state,
            )

        passed = False
        try:
            passed = bool(self._run_eval_gate(start_state=start_state))
        except Exception as e:
            if self.verbose:
                print(
                    "[level-gate] eval failed:",
                    f"{type(e).__name__}: {e}",
                )
            passed = False

        self.logger.record("custom/level_gate_eval_triggered", 1)
        self.logger.record("custom/level_gate_eval_passed", 1 if passed else 0)

        if passed:
            meta = {
                "source": "LevelGateEvalCallback",
                "from_state": start_state,
                "required_successes": int(self.required_successes),
                "num_timesteps": int(self.num_timesteps),
            }
            self._write_next_state(next_state=next_state, meta=meta)
            if self.verbose:
                print("[level-gate] committed next_state:", next_state)

        return True
