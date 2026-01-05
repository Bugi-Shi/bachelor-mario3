from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class ResetStatsCallback(BaseCallback):
    """Count and periodically print why episodes ended.

    Tracks total episode ends, plus how many were caused by:
    - `no_hpos_progress_terminated` (stuck)
    - `life_lost_terminated` (life lost)
    """

    def __init__(self, print_every_steps: int = 5000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_every_steps = int(print_every_steps)
        self.total_dones = 0
        self.stuck_dones = 0
        self.life_lost_dones = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is not None and dones is not None:
            for episode_done, info in zip(dones, infos):
                if not episode_done or not isinstance(info, dict):
                    continue
                self.total_dones += 1
                if info.get("no_hpos_progress_terminated"):
                    self.stuck_dones += 1
                if info.get("life_lost_terminated"):
                    self.life_lost_dones += 1

        if self.print_every_steps and (
            self.n_calls % self.print_every_steps == 0
        ):
            print(
                "[reset-stats] dones=",
                self.total_dones,
                "stuck=",
                self.stuck_dones,
                "life_lost=",
                self.life_lost_dones,
            )
        return True
