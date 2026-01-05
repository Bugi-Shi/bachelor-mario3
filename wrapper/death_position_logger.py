from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import gymnasium as gym

from utils import to_python_int_or_none


class DeathPositionLoggerWrapper(gym.Wrapper):
    """Logs (approx) death positions as JSONL.

    Logging happens on steps that end an episode where either:
    - info['life_lost_terminated'] is True (or lives dropped)
    - info['no_hpos_progress_terminated'] is True (optional, still logged)

    Output schema (one JSON object per death):
    - ep: per-env episode counter (starts at 1)
    - reason: "life_lost" | "stuck"
    - x: global x (the same value exposed as info['x'] and used by videos)
    - level: short level code (e.g. "1-1", "1-3", "1-A", "1-MF")

        Preferred position signal:
        - If Retro provides `info['world_x_hi']` (a page/screen counter byte),
            we compute global x directly as:

                    global_x = world_x_hi * screen_width + hpos

        Fallback:
        - If `world_x_hi` is unavailable, we fall back to a heuristic that
            detects hpos "wraps" and increments an internal screen index.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        log_path: str,
        cancel_life_loss_penalty_after_x: Optional[int] = None,
        life_loss_penalty_value: float = 100.0,
        screen_width: int = 256,
        wrap_threshold: int = 50,
        wrap_prev_min: int = 200,
        wrap_cur_max: int = 60,
        wrap_cooldown_steps: int = 15,
        max_world_x_hi_delta: int = 2,
    ):
        super().__init__(env)
        self.log_path = Path(log_path)
        self.cancel_life_loss_penalty_after_x = (
            int(cancel_life_loss_penalty_after_x)
            if cancel_life_loss_penalty_after_x is not None
            else None
        )
        self.life_loss_penalty_value = float(life_loss_penalty_value)
        self.screen_width = int(screen_width)
        self.wrap_threshold = int(wrap_threshold)
        self.wrap_prev_min = int(wrap_prev_min)
        self.wrap_cur_max = int(wrap_cur_max)
        self.wrap_cooldown_steps = int(wrap_cooldown_steps)
        self.max_world_x_hi_delta = int(max_world_x_hi_delta)

        self._prev_hpos: Optional[int] = None
        self._prev_lives: Optional[int] = None
        self._screen_idx: int = 0
        self._wrap_cooldown: int = 0
        self._last_global_x: Optional[int] = None
        self._last_hpos: Optional[int] = None
        self._logged_this_episode: bool = False
        self._episode_idx: int = 0
        self._episode_num: int = 0

    def reset(self, **kwargs):
        self._episode_idx += 1
        self._episode_num = int(self._episode_idx)
        self._prev_hpos = None
        self._prev_lives = None
        self._screen_idx = 0
        self._wrap_cooldown = 0
        self._last_global_x = None
        self._last_hpos = None
        self._logged_this_episode = False
        return self.env.reset(**kwargs)

    @staticmethod
    def _level_code(raw_state: Optional[object]) -> str:
        """Return a short level code like '1-1' / '1-A' / '1-MF'."""

        if raw_state is None:
            return "Unknown"

        s = str(raw_state).strip()
        if not s:
            return "Unknown"

        name = Path(s).name
        if name.endswith(".state"):
            name = name[: -len(".state")]
        if name.startswith("1Player."):
            name = name[len("1Player."):]

        # Expect patterns like:
        #   World1.Level3
        #   World1.Airship
        #   World1.MiniFortress
        # but be tolerant to other variants.
        import re

        m_world = re.search(r"World(?P<w>\d+)", name)
        world = m_world.group("w") if m_world else "1"

        if re.search(r"Airship", name, flags=re.IGNORECASE):
            return f"{world}-A"
        if re.search(r"MiniFortress", name, flags=re.IGNORECASE):
            return f"{world}-MF"

        m_level = re.search(r"Level(?P<l>\d+)", name)
        if m_level:
            return f"{world}-{m_level.group('l')}"

        return "Unknown"

    def _choose_screen_idx(
        self,
        *,
        cur_hpos: Optional[int],
        cur_world_x_hi: Optional[int],
        ended: bool,
    ) -> Optional[int]:
        """Return the best-available screen index for computing global X.

        Strategy:
        - Always keep the wrap-based heuristic updated.
        - Prefer cur_world_x_hi only if it is close to the heuristic screen
          index (guards against occasional bogus RAM reads).
        """

        self._maybe_advance_screen(cur_hpos, ended=ended)

        if cur_hpos is None:
            return None

        if cur_world_x_hi is None:
            return int(self._screen_idx)

        if (
            abs(int(cur_world_x_hi) - int(self._screen_idx))
            <= self.max_world_x_hi_delta
        ):
            # Sync heuristic to RAM page counter when it looks sane.
            self._screen_idx = int(cur_world_x_hi)
            return int(cur_world_x_hi)

        # RAM value looks implausible; stick to heuristic.
        return int(self._screen_idx)

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

        cur_hpos = to_python_int_or_none(info.get("hpos"))
        cur_lives = to_python_int_or_none(info.get("lives"))
        cur_world_x_hi = to_python_int_or_none(info.get("world_x_hi"))

        life_lost_by_lives = False
        if cur_lives is not None:
            if self._prev_lives is not None and cur_lives < self._prev_lives:
                life_lost_by_lives = True
            self._prev_lives = cur_lives

        chosen_screen_idx = self._choose_screen_idx(
            cur_hpos=cur_hpos, cur_world_x_hi=cur_world_x_hi, ended=ended
        )

        # Expose derived position info to downstream consumers.
        if cur_hpos is not None and chosen_screen_idx is not None:
            info = dict(info)
            info["screen_idx"] = int(chosen_screen_idx)
            info["x"] = int(
                int(chosen_screen_idx) * self.screen_width + int(cur_hpos)
            )

        # Track last meaningful position BEFORE terminal frames.
        # On life loss the last frame often has hpos=0 (respawn), which we
        # don't want to overwrite.
        if not ended and cur_hpos is not None:
            self._last_hpos = int(cur_hpos)
            if chosen_screen_idx is not None:
                self._last_global_x = int(
                    int(chosen_screen_idx) * self.screen_width + int(cur_hpos)
                )

        # Decide whether to log and why.
        reason = None
        if info.get("life_lost_terminated") or life_lost_by_lives:
            reason = "life_lost"
        elif info.get("no_hpos_progress_terminated"):
            reason = "stuck"

        # Terminal frames often have a bogus position signal (e.g. hpos=0).
        # Prefer the last known pre-terminal global x for downstream logic.
        x_pre_terminal = self._last_global_x
        if ended and x_pre_terminal is not None:
            info = dict(info)
            info["x"] = int(x_pre_terminal)

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

            # Fallbacks if we never captured a pre-terminal position.
            if global_x is None and cur_hpos is not None:
                global_x = int(
                    self._screen_idx * self.screen_width + int(cur_hpos)
                )
            if global_x is None:
                global_x = 0

            level = self._level_code(info.get("episode_state"))
            self._write(
                {
                    "ep": int(self._episode_num),
                    "reason": reason,
                    "x": int(global_x),
                    "level": level,
                }
            )
            self._logged_this_episode = True

        return obs, reward, terminated, truncated, info
