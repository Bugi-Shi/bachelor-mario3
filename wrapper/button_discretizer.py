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

    def __init__(self, env):
        super().__init__(env)
        button_to_idx = {b: i for i, b in enumerate(self.BUTTONS)}

        self._actions = []
        for action_buttons in self.ACTIONS:
            arr = np.zeros((len(self.BUTTONS),), dtype=bool)
            for button in action_buttons:
                arr[button_to_idx[button]] = True
            self._actions.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a]
