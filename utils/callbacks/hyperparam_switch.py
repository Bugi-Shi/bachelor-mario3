from __future__ import annotations

from typing import Iterable, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn


def _normalize_state_name(state: object) -> str:
    s = str(state).strip()
    if not s:
        return ""

    # Accept both plain names and full paths to *.state.
    # Keep it consistent with wrappers.
    import pathlib

    name = pathlib.Path(s).name
    if name.endswith(".state"):
        name = name[: -len(".state")]
    return name


class HyperparamSwitchOnLevelCallback(BaseCallback):
    """Switch PPO hyperparameters once a target level is reached.

    Intended use: increase exploration for a known bottleneck (e.g. Level 1-2
    pit and 1-6) by raising ent_coef and lowering learning_rate
    as soon as episodes start from that level.

    Notes:
        - This updates PPO attributes in-place and rewires lr_schedule so
            Stable-Baselines3 doesn't overwrite optimizer lrs on the next
            update.
    - The switch is one-way by default (no revert).
    """

    def __init__(
        self,
        *,
        trigger_episode_states: Iterable[str],
        revert_episode_states: Iterable[str] = (),
        ent_coef_after: float,
        learning_rate_after: float,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._trigger_states = {
            _normalize_state_name(s) for s in trigger_episode_states
        }
        self._revert_states = {
            _normalize_state_name(s) for s in revert_episode_states
        }
        self.ent_coef_after = float(ent_coef_after)
        self.learning_rate_after = float(learning_rate_after)

        # True while we're actively using the "after" hyperparams.
        self._active: bool = False
        self._ent_coef_before: Optional[float] = None
        self._learning_rate_before: Optional[float] = None

    def _init_callback(self) -> None:
        # Snapshot current settings from the model.
        try:
            self._ent_coef_before = float(getattr(self.model, "ent_coef"))
        except Exception:
            self._ent_coef_before = None
        try:
            self._learning_rate_before = float(
                getattr(self.model, "learning_rate")
            )
        except Exception:
            self._learning_rate_before = None

    def _apply(
        self,
        *,
        ent_coef: Optional[float],
        learning_rate: Optional[float],
    ) -> None:
        if ent_coef is not None:
            try:
                setattr(self.model, "ent_coef", float(ent_coef))
            except Exception:
                pass

        if learning_rate is None:
            return

        try:
            new_lr = float(learning_rate)
            setattr(self.model, "learning_rate", new_lr)
            # Ensure SB3's internal update_learning_rate uses the new constant.
            setattr(self.model, "lr_schedule", get_schedule_fn(new_lr))
        except Exception:
            new_lr = None

        if new_lr is None:
            return

        try:
            opt = getattr(self.model.policy, "optimizer", None)
            if opt is not None:
                for group in opt.param_groups:
                    group["lr"] = float(new_lr)
        except Exception:
            pass

    def _maybe_switch(self, *, episode_state: str) -> None:
        if self._active:
            return
        if not episode_state or episode_state not in self._trigger_states:
            return

        self._apply(
            ent_coef=float(self.ent_coef_after),
            learning_rate=float(self.learning_rate_after),
        )
        self._active = True

        # Emit a one-time console notice and a TB scalar.
        if self.verbose:
            print(
                "[hparam-switch] Switched on episode_state=",
                episode_state,
                "ent_coef:",
                self._ent_coef_before,
                "->",
                self.ent_coef_after,
                "lr:",
                self._learning_rate_before,
                "->",
                self.learning_rate_after,
            )

        try:
            self.logger.record("custom/hparam_switched", 1)
            self.logger.record("custom/ent_coef", float(self.ent_coef_after))
            self.logger.record(
                "custom/learning_rate", float(self.learning_rate_after)
            )
        except Exception:
            pass

    def _maybe_revert(self, *, episode_state: str) -> None:
        if not self._active:
            return
        if not self._revert_states:
            return
        if not episode_state or episode_state not in self._revert_states:
            return

        self._apply(
            ent_coef=self._ent_coef_before,
            learning_rate=self._learning_rate_before,
        )
        self._active = False

        if self.verbose:
            print(
                "[hparam-switch] Reverted on episode_state=",
                episode_state,
                "ent_coef:",
                getattr(self.model, "ent_coef", None),
                "lr:",
                getattr(self.model, "learning_rate", None),
            )

        try:
            self.logger.record("custom/hparam_reverted", 1)
            if self._ent_coef_before is not None:
                self.logger.record(
                    "custom/ent_coef", float(self._ent_coef_before)
                )
            if self._learning_rate_before is not None:
                self.logger.record(
                    "custom/learning_rate",
                    float(self._learning_rate_before),
                )
        except Exception:
            pass

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue
            raw = info.get("episode_state")
            if raw is None:
                continue
            state = _normalize_state_name(raw)
            self._maybe_switch(episode_state=state)
            self._maybe_revert(episode_state=state)

        return True
