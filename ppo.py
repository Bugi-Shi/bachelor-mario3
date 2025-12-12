# Globbal imports
import retro
import retro.data as retro_data
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

# Local imports
from utils.buttondiscretizer import ButtonDiscretizerWrapper
from utils.noprogesspenalty import NoHposProgressGuardWrapper
from utils.resetbydeath import ResetToDefaultStateByDeathWrapper


class ResetStatsCallback(BaseCallback):
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
            for done, info in zip(dones, infos):
                if not done:
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
                "[reset-stats] dones=", self.total_dones,
                "stuck=", self.stuck_dones,
                "life_lost=", self.life_lost_dones,
            )
        return True


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


def make_mario_env(custom_data_root: str):
    """Factory for SubprocVecEnv (must be picklable for spawn)."""
    retro_data.add_custom_integration(custom_data_root)
    default_state = _read_default_state(custom_data_root)
    env = retro.make(
        "SuperMarioBros3-Nes",
        inttype=retro_data.Integrations.CUSTOM_ONLY,
    )
    env = ResetToDefaultStateByDeathWrapper(
        env,
        default_state=default_state,
        inttype=retro_data.Integrations.CUSTOM_ONLY,
        lives_key="lives",
    )
    env = NoHposProgressGuardWrapper(env)
    env = ButtonDiscretizerWrapper(env)
    return env


def train_ppo(path: str = ""):
    """
    Simple PPO training using SuperMarioBros3-Nes-v0 with VecFrameStack.
    """
    max_steps = 1000000
    n_steps = 512
    n_stack = 4
    n_envs = 4
    custom_data_root = str(Path(__file__).resolve().parent / "retro_custom")

    venv = SubprocVecEnv(
        [lambda: make_mario_env(custom_data_root) for _ in range(n_envs)]
    )

    venv = VecFrameStack(venv, n_stack=n_stack)

    if (path):
        model = PPO.load(path)
        model.set_env(venv)
    else:
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            device="cuda",
            n_steps=n_steps,
            verbose=1,
        )

    model.learn(total_timesteps=max_steps, callback=ResetStatsCallback())
    model.save("ppo_super_mario_bros3")
    model = PPO.load("ppo_super_mario_bros3")
    model.set_env(venv)
    obs = venv.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)  # type: ignore[arg-type]
        obs, rewards, dones, infos = venv.step(action)
        print(infos)
        done = dones.any()
        venv.render()
    venv.close()
