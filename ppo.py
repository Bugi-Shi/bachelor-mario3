import numpy as np
import retro
import retro.data as retro_data
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from pathlib import Path


class NoProgressPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, info_key: str = "hpos", penalty: float = 0.5):
        super().__init__(env)
        self.info_key = info_key
        self.penalty = float(penalty)
        self._prev_value = None

    def reset(self, **kwargs):
        self._prev_value = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cur = info.get(self.info_key)
        if cur is not None:
            try:
                cur_val = int(cur)
            except Exception:
                cur_val = int(np.asarray(cur).item())

            if self._prev_value is not None and cur_val == self._prev_value:
                reward = float(reward) - self.penalty

            self._prev_value = cur_val
        return obs, reward, terminated, truncated, info


def train_ppo():
    """
    Simple PPO training using SuperMarioBros3-Nes-v0 with VecFrameStack.
    """
    max_steps = 10000
    n_steps = 128
    n_stack = 4

    # Use workspace-local Retro game data (ROM + states + json)
    # instead of the installed dataset.
    custom_data_root = Path(__file__).resolve().parent / "retro_custom"
    retro_data.add_custom_integration(str(custom_data_root))
    env = retro.make(
        "SuperMarioBros3-Nes",
        inttype=retro_data.Integrations.CUSTOM_ONLY,
    )

    # Penalize standing still (no progress in hpos).
    env = NoProgressPenaltyWrapper(env, info_key="hpos", penalty=0.5)

    venv = DummyVecEnv([lambda: env])
    venv = VecFrameStack(venv, n_stack=n_stack)

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=n_steps,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=max_steps)
    model.save("ppo_super_mario_bros3")
    model = PPO.load("ppo_super_mario_bros3")
    model.set_env(venv)
