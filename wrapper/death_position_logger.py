from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import gymnasium as gym
from utils import to_python_int_or_none


class DeathPositionLoggerWrapper(gym.Wrapper):
    """Logs death/stuck positions as JSONL and exposes a derived `info['x']`.

    Minimal behavior:
    - Derive a global X coordinate as: `x = world_x_hi * screen_width + hpos`
      when those RAM signals exist.
        - Fallback (if `world_x_hi` is missing): approximate a screen index by
            detecting large `hpos` wraps.
    - On terminal frames, prefer the last non-terminal `x`.
    - If a death/stuck event is detected, append a JSONL record.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        log_path: str,
        level_label: str = "1-1",
        cancel_life_loss_penalty_after_x: Optional[int] = None,
        life_loss_penalty_value: float = 100.0,
        screen_width: int = 256,
    ):
        super().__init__(env)
        self.log_path = Path(log_path)
        self.level_label = str(level_label).strip() or "1-1"
        self.cancel_life_loss_penalty_after_x = (
            int(cancel_life_loss_penalty_after_x)
            if cancel_life_loss_penalty_after_x is not None
            else None
        )
        self.life_loss_penalty_value = float(life_loss_penalty_value)
        self.screen_width = int(screen_width)

        self._prev_hpos: Optional[int] = None
        self._screen_idx: int = 0
        self._prev_lives: Optional[int] = None
        self._last_global_x: Optional[int] = None
        self._logged_this_episode: bool = False
        self._episode_idx: int = 0
        self._episode_num: int = 0

    def reset(self, **kwargs):
        self._episode_idx += 1
        self._episode_num = int(self._episode_idx)
        self._prev_hpos = None
        self._screen_idx = 0
        self._prev_lives = None
        self._last_global_x = None
        self._logged_this_episode = False
        return self.env.reset(**kwargs)

    def _compute_global_x(self, info: dict, *, ended: bool) -> Optional[int]:
        hpos = to_python_int_or_none(info.get("hpos"))
        world_x_hi = to_python_int_or_none(info.get("world_x_hi"))
        if hpos is None:
            return None

        # Preferred: use Retro's page counter when available.
        if world_x_hi is not None:
            if not ended:
                self._screen_idx = int(world_x_hi)
                self._prev_hpos = int(hpos)
            return int(int(world_x_hi) * int(self.screen_width) + int(hpos))

        # Fallback: approximate page transitions by detecting hpos wraps.
        # Kept intentionally simple (good enough as a backup signal).
        if not ended:
            if self._prev_hpos is not None:
                prev = int(self._prev_hpos)
                cur = int(hpos)
                if cur + 50 < prev and prev >= 200 and cur <= 60:
                    self._screen_idx += 1
            self._prev_hpos = int(hpos)

        return int(int(self._screen_idx) * int(self.screen_width) + int(hpos))

    def _write(self, obj: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ended = bool(terminated or truncated)

        cur_lives = to_python_int_or_none(info.get("lives"))

        life_lost_by_lives = False
        if cur_lives is not None:
            if self._prev_lives is not None and cur_lives < self._prev_lives:
                life_lost_by_lives = True
            self._prev_lives = cur_lives

        info_out = dict(info)
        cur_global_x = self._compute_global_x(info_out, ended=ended)
        if cur_global_x is not None:
            info_out["x"] = int(cur_global_x)

        # Track last meaningful position BEFORE terminal frames.
        # (Terminal frames can have bogus values after respawn.)
        if not ended and cur_global_x is not None:
            self._last_global_x = int(cur_global_x)

        # Decide whether to log and why.
        reason = None
        if info_out.get("life_lost_terminated") or life_lost_by_lives:
            reason = "life_lost"
        elif info_out.get("no_hpos_progress_terminated"):
            reason = "stuck"

        # Terminal frames often have a bogus position signal (e.g. hpos=0).
        # Prefer the last known pre-terminal global x for downstream logic.
        x_pre_terminal = self._last_global_x
        if ended and x_pre_terminal is not None:
            info_out["x"] = int(x_pre_terminal)

        # Optionally cancel the Retro scenario life-loss penalty after
        # reaching a certain progress threshold.
        if (
            reason == "life_lost"
            and self.cancel_life_loss_penalty_after_x is not None
            and x_pre_terminal is not None
            and int(x_pre_terminal)
            > int(self.cancel_life_loss_penalty_after_x)
        ):
            reward = float(reward) + float(self.life_loss_penalty_value)

        should_log = bool(
            reason is not None and (ended or reason == "life_lost")
        )
        if should_log and not self._logged_this_episode:
            # Prefer last known pre-terminal position.
            global_x = self._last_global_x

            # Fallback: best-effort current x, otherwise 0.
            if global_x is None:
                global_x = cur_global_x
            if global_x is None:
                global_x = 0

            self._write(
                {
                    "ep": int(self._episode_num),
                    "reason": reason,
                    "x": int(global_x),
                    "level": self.level_label,
                }
            )
            self._logged_this_episode = True

        return obs, reward, terminated, truncated, info_out
