import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np


class DeathPositionLoggerWrapper(gym.Wrapper):
    """Logs (approx) death positions as JSONL.

    Logging happens on steps that end an episode where either:
    - info['life_lost_terminated'] is True
    - info['no_hpos_progress_terminated'] is True (optional, still logged)

    Because SMB3 hpos in this integration is currently a single byte, it can
    "wrap" back to 0. We track a screen index and compute a global x as:

        global_x = screen_idx * screen_width + hpos

    This makes it easy to overlay points on a stitched level image.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        log_path: str,
        screen_width: int = 256,
        wrap_threshold: int = 50,
        wrap_prev_min: int = 200,
        wrap_cur_max: int = 60,
        wrap_cooldown_steps: int = 15,
    ):
        super().__init__(env)
        self.log_path = Path(log_path)
        self.screen_width = int(screen_width)
        self.wrap_threshold = int(wrap_threshold)
        self.wrap_prev_min = int(wrap_prev_min)
        self.wrap_cur_max = int(wrap_cur_max)
        self.wrap_cooldown_steps = int(wrap_cooldown_steps)

        self._prev_hpos: Optional[int] = None
        self._prev_lives: Optional[int] = None
        self._screen_idx: int = 0
        self._wrap_cooldown: int = 0
        self._last_global_x: Optional[int] = None
        self._last_hpos: Optional[int] = None
        self._logged_this_episode: bool = False

    def reset(self, **kwargs):
        self._prev_hpos = None
        self._prev_lives = None
        self._screen_idx = 0
        self._wrap_cooldown = 0
        self._last_global_x = None
        self._last_hpos = None
        self._logged_this_episode = False
        return self.env.reset(**kwargs)

    @staticmethod
    def _to_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return int(np.asarray(value).item())

    def _maybe_advance_screen(
        self, cur_hpos: Optional[int], ended: bool
    ) -> None:
        if ended or cur_hpos is None:
            return

        if self._wrap_cooldown > 0:
            self._wrap_cooldown -= 1

        if self._prev_hpos is None:
            self._prev_hpos = cur_hpos
            return
        # Heuristic: large drop without episode end -> likely screen boundary.
        if (
            self._wrap_cooldown == 0
            and cur_hpos + self.wrap_threshold < self._prev_hpos
            and self._prev_hpos >= self.wrap_prev_min
            and cur_hpos <= self.wrap_cur_max
        ):
            self._screen_idx += 1
            self._wrap_cooldown = self.wrap_cooldown_steps
        self._prev_hpos = cur_hpos

    def _write(self, obj: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ended = bool(terminated or truncated)

        cur_hpos = self._to_int(info.get("hpos"))
        cur_lives = self._to_int(info.get("lives"))

        life_lost_by_lives = False
        if cur_lives is not None:
            if self._prev_lives is not None and cur_lives < self._prev_lives:
                life_lost_by_lives = True
            self._prev_lives = cur_lives

        self._maybe_advance_screen(cur_hpos, ended=ended)

        # Track last meaningful position BEFORE terminal frames.
        # On life loss the last frame often has hpos=0 (respawn), which we
        # don't want to overwrite.
        if not ended and cur_hpos is not None:
            self._last_hpos = int(cur_hpos)
            self._last_global_x = int(
                self._screen_idx * self.screen_width + int(cur_hpos)
            )

        # Decide whether to log and why.
        reason = None
        if info.get("life_lost_terminated") or life_lost_by_lives:
            reason = "life_lost"
        elif info.get("no_hpos_progress_terminated"):
            reason = "stuck"

        should_log = bool(
            reason is not None and (ended or reason == "life_lost")
        )
        if should_log and not self._logged_this_episode:
            # Prefer last known pre-terminal position.
            global_x = self._last_global_x
            logged_hpos = self._last_hpos

            # Fallbacks if we never captured a pre-terminal position.
            if global_x is None and cur_hpos is not None:
                logged_hpos = int(cur_hpos)
                global_x = int(
                    self._screen_idx * self.screen_width + int(cur_hpos)
                )
            if global_x is None:
                logged_hpos = 0
                global_x = 0

            hpos_raw = None if cur_hpos is None else int(cur_hpos)
            self._write(
                {
                    "reason": reason,
                    "x": int(global_x),
                    "y": None,
                    "hpos": int(logged_hpos) if logged_hpos is not None else 0,
                    "lives": None if cur_lives is None else int(cur_lives),
                    "hpos_raw": hpos_raw,
                    "screen_idx": int(self._screen_idx),
                }
            )
            self._logged_this_episode = True

        return obs, reward, terminated, truncated, info
