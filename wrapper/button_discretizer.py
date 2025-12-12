import gymnasium as gym
import numpy as np


class ButtonDiscretizerWrapper(gym.ActionWrapper):
    """
    Wrap a gym retro environment and make it use discrete
    actions for the NES controller.

    NES Button Layout:
    ['B', 'NULL', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    """

    def __init__(self, env):
        super(ButtonDiscretizerWrapper, self).__init__(env)
        buttons = ['B',
                   'NULL',
                   'SELECT',
                   'START',
                   'UP',
                   'DOWN',
                   'LEFT',
                   'RIGHT',
                   'A']

        actions = [
            ['RIGHT'],                  # Move Right
            ['RIGHT', 'A'],             # Move Right + Jump
            ['RIGHT', 'B'],             # Move Right + Run
            ['RIGHT', 'B', 'A'],        # Move Right + Run + Jump
            ['A'],                      # Jump
            ['LEFT'],                   # Move Left
            ['LEFT', 'A'],              # Move Left + Jump
            ['LEFT', 'B'],              # Move Left + Run
            ['LEFT', 'B', 'A'],         # Move Left + Run + Jump
            ['UP'],                     # Climb Up
            ['DOWN'],                   # Duck
            []                          # No Action
        ]
        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a]
