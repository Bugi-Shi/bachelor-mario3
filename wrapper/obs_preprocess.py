import gymnasium as gym
import numpy as np
from PIL import Image


class GrayscaleResizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = int(width)
        self.height = int(height)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8,
        )

    def observation(self, observation):
        obs = np.asarray(observation)

        if obs.dtype != np.uint8:
            if np.issubdtype(obs.dtype, np.floating):
                if obs.size and obs.max() <= 1.0:
                    obs = np.clip(obs * 255.0, 0, 255)
                else:
                    obs = np.clip(obs, 0, 255)
            obs = obs.astype(np.uint8)

        if obs.ndim == 3 and obs.shape[2] == 1:
            gray = obs[:, :, 0]
        elif obs.ndim == 3:
            gray_img = Image.fromarray(obs).convert("L")
            gray = np.asarray(gray_img, dtype=np.uint8)
        elif obs.ndim == 2:
            gray = obs
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")

        resampling = getattr(Image, "Resampling", None)
        if resampling is not None:
            resample = resampling.BILINEAR
        else:
            resample = getattr(Image, "BILINEAR")

        img = Image.fromarray(gray, mode="L").resize(
            (self.width, self.height),
            resample=resample,
        )
        out = np.asarray(img, dtype=np.uint8)
        return out[:, :, None]
