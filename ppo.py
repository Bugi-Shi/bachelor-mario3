# Globbal imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

# Local imports
from env.MB3_env import mariobros3_env


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


def train_ppo():
    """
    Simple PPO training using SuperMarioBros3-Nes-v0 with VecFrameStack.
    """
    max_steps = 10000
    batch_size = 64
    n_steps = 2048
    n_stack = 4
    n_envs = 1
    custom_data_root = str(Path(__file__).resolve().parent / "retro_custom")

    venv = SubprocVecEnv(
        [
            lambda i=i: mariobros3_env(custom_data_root, rank=i)
            for i in range(n_envs)
        ]
    )

    venv = VecFrameStack(venv, n_stack=n_stack)

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        batch_size=batch_size,
        device="cuda",
        ent_coef=0.01,
        n_epochs=3,
        n_steps=n_steps,
        verbose=1,
        learning_rate=2.5e-4,
        clip_range=0.2,
    )

    model.learn(total_timesteps=max_steps, callback=ResetStatsCallback())
    model.save("ppo_super_mario_bros3")
    model = PPO.load("ppo_super_mario_bros3", device="cuda")
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
