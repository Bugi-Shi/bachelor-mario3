from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image


class GrayscaleResizeObservationWrapper(gym.ObservationWrapper):
    """Convert observations to grayscale and resize to `(height, width, 1)`.

    Accepts:
    - RGB `uint8` frames (H, W, 3)
    - grayscale frames (H, W) or (H, W, 1)
    - float frames (commonly in [0, 1]) which get scaled to [0, 255]
    """

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
        # For example, 0.5 is grey in float [0, 1]
        # Normalize dtype to uint8 in [0, 255] for 8-bit images
        obs = np.asarray(observation)
        if obs.dtype != np.uint8:
            if np.issubdtype(obs.dtype, np.floating):
                # Common case: floats in [0, 1].
                max_val = float(obs.max()) if obs.size else 0.0
                if max_val <= 1.0:
                    obs = obs * 255.0
            obs = np.clip(obs, 0, 255).astype(np.uint8)

        # Convert to a 2D grayscale image.
        if obs.ndim == 2:
            gray_2d = obs
        elif obs.ndim == 3 and obs.shape[2] == 1:
            gray_2d = obs[:, :, 0]
        elif obs.ndim == 3 and obs.shape[2] == 3:
            gray_2d = np.asarray(
                Image.fromarray(obs).convert("L"), dtype=np.uint8
            )
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")

        # Resize and add the channel dimension back (H, W, 1).
        resized = Image.fromarray(gray_2d, mode="L").resize(
            (self.width, self.height),
            resample=_pil_bilinear_resample(),
        )
        resized_gray = np.asarray(resized, dtype=np.uint8)
        return resized_gray[:, :, None]


def _pil_bilinear_resample() -> int:
    """Return the PIL bilinear resampling enum across Pillow versions."""

    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.BILINEAR)
    return int(getattr(Image, "BILINEAR"))
