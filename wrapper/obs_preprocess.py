from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image


class GrayscaleResizeObservationWrapper(gym.ObservationWrapper):
    """Converts RGB observations to grayscale and resizes to (H, W, 1)."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width: int = int(width)
        self.height: int = int(height)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8,
        )

    def observation(self, observation: Any) -> np.ndarray:
        obs = np.asarray(observation)

        if obs.dtype != np.uint8:
            if np.issubdtype(obs.dtype, np.floating):
                if obs.size and obs.max() <= 1.0:
                    obs = np.clip(obs * 255.0, 0, 255)
                else:
                    obs = np.clip(obs, 0, 255)
            obs = obs.astype(np.uint8)

        if obs.ndim == 3 and obs.shape[2] == 1:
            gray_2d = obs[:, :, 0]
        elif obs.ndim == 3:
            gray_img = Image.fromarray(obs).convert("L")
            gray_2d = np.asarray(gray_img, dtype=np.uint8)
        elif obs.ndim == 2:
            gray_2d = obs
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")

        img = Image.fromarray(gray_2d, mode="L").resize(
            (self.width, self.height),
            resample=_pil_bilinear_resample(),
        )
        resized_gray = np.asarray(img, dtype=np.uint8)
        return resized_gray[:, :, None]


def _pil_bilinear_resample() -> int:
    """Return the PIL bilinear resampling enum across Pillow versions."""

    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.BILINEAR)
    return int(getattr(Image, "BILINEAR"))
