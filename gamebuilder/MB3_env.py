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
):
    frame_skip = 4
    rank = int(rank)

    retro_data.add_custom_integration(custom_data_root)
    default_state = _read_default_state(custom_data_root)
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

    # Keep "~4 seconds no progress" consistent even with frame_skip.
    env = NoHposProgressGuardWrapper(
        env,
        max_no_progress_steps=max(1, 240 // frame_skip),
    )
    env = ButtonDiscretizerWrapper(env)

    # Log death positions for plotting (per-process file).
    # This wrapper must be OUTERMOST so it can observe terminations set by
    # any other wrapper (life loss, stuck termination, etc.).
    if run_dir:
        deaths_dir = Path(run_dir) / "deaths"
    else:
        project_root = Path(custom_data_root).resolve().parent
        deaths_dir = project_root / "outputs" / "deaths"
    log_path = deaths_dir / f"deaths_env{rank}.jsonl"
    env = DeathPositionLoggerWrapper(env, log_path=str(log_path))
    return env
