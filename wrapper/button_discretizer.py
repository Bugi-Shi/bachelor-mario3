import gymnasium as gym
import numpy as np


class ButtonDiscretizerWrapper(gym.ActionWrapper):
    """
    Wrap a gym retro environment and make it use discrete
    actions for the NES controller.

    NES Button Layout:
    ['B', 'NULL', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    """

    BUTTONS = (
        "B",
        "NULL",
        "SELECT",
        "START",
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "A",
    )

    ACTIONS = (
        ("RIGHT",),
        ("RIGHT", "A"),
        ("RIGHT", "B"),
        ("RIGHT", "B", "A"),
        ("A",),
        ("LEFT",),
        ("LEFT", "A"),
        ("LEFT", "B"),
        ("LEFT", "B", "A"),
        ("UP",),
        ("DOWN",),
        tuple(),
    )

    def __init__(self, env: gym.Env):
        super().__init__(env)
        button_to_index = {name: i for i, name in enumerate(self.BUTTONS)}

        self._button_masks: list[np.ndarray] = []
        for pressed_buttons in self.ACTIONS:
            mask = np.zeros((len(self.BUTTONS),), dtype=bool)
            for button_name in pressed_buttons:
                mask[button_to_index[button_name]] = True
            self._button_masks.append(mask)

        self.action_space = gym.spaces.Discrete(len(self._button_masks))

    def action(self, action_index: int) -> np.ndarray:
        return self._button_masks[action_index]
