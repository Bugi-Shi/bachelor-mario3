import json
from pathlib import Path
from typing import Optional

import retro
import retro.data as retro_data
from gymnasium.wrappers import MaxAndSkipObservation

# Local imports
from wrapper.button_discretizer import ButtonDiscretizerWrapper
from wrapper.no_progess_penalty import NoHposProgressGuardWrapper
from wrapper.reset_by_death import ResetToDefaultStateByDeathWrapper
from wrapper.obs_preprocess import GrayscaleResizeObservationWrapper
from wrapper.death_position_logger import DeathPositionLoggerWrapper
from wrapper.goal_reward_and_state_switch import (
    GoalRewardAndStateSwitchWrapper,
)


def _read_default_state(custom_data_root: str) -> str:
    metadata_path = (
        Path(custom_data_root) / "SuperMarioBros3-Nes" / "metadata.json"
    )
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    default_state = metadata.get("default_state")
    if not default_state or not isinstance(default_state, str):
        raise ValueError("metadata.json missing a valid 'default_state'")
    return default_state


def mariobros3_env(
    custom_data_root: str,
    rank: int = 0,
    *,
    run_dir: Optional[str] = None,
    enable_death_logger: bool = True,
    render_mode: Optional[str] = None,
):
    frame_skip = 4
    rank = int(rank)

    retro_data.add_custom_integration(custom_data_root)
    default_state = _read_default_state(custom_data_root)
    try:
        env = retro.make(
            "SuperMarioBros3-Nes",
            inttype=retro_data.Integrations.CUSTOM_ONLY,
            render_mode=render_mode,
        )
    except TypeError:
        # Some retro versions don't expose render_mode in the make() API.
        env = retro.make(
            "SuperMarioBros3-Nes",
            inttype=retro_data.Integrations.CUSTOM_ONLY,
        )

    env = MaxAndSkipObservation(env, skip=frame_skip)

    env = GrayscaleResizeObservationWrapper(env, width=84, height=84)

    env = ResetToDefaultStateByDeathWrapper(
        env,
        default_state=default_state,
        inttype=retro_data.Integrations.CUSTOM_ONLY,
        lives_key="lives",
    )

    # Keep "~8 seconds no progress" consistent even with frame_skip.
    env = NoHposProgressGuardWrapper(
        env,
        max_no_progress_steps=max(1, 480 // frame_skip),
    )
    env = ButtonDiscretizerWrapper(env)

    # Log death positions for plotting (per-process file).
    # This wrapper must be OUTERMOST so it can observe terminations set by
    # any other wrapper (life loss, stuck termination, etc.).
    if enable_death_logger:
        if run_dir:
            deaths_dir = Path(run_dir) / "deaths"
        else:
            project_root = Path(custom_data_root).resolve().parent
            deaths_dir = project_root / "outputs" / "deaths"
        log_path = deaths_dir / f"deaths_env{rank}.jsonl"
        env = DeathPositionLoggerWrapper(env, log_path=str(log_path))

    # Big reward at the end-of-level goal and switch future resets to Level 3.
    if run_dir:
        shared_switch_path = str(Path(run_dir) / "level_switch.json")
    else:
        project_root = Path(custom_data_root).resolve().parent
        shared_switch_path = str(
            project_root / "outputs" / "level_switch.json"
        )

    env = GoalRewardAndStateSwitchWrapper(
        env,
        goal_x=2685,
        goal_reward=500.0,
        next_state="1Player.World1.Level3",
        shared_switch_path=shared_switch_path,
    )
    return env
