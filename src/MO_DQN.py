import gymnasium as gym
from mo_gymnasium import mo_gym
from torch import device
from torch._C import device

class MO_DQN(MOAgent, MOPolicy):
    """ Implements multi-objective DQN working with one agent"""

    def __init__(self, env: gym.Env | None, device: device | str = "auto", seed: int | None = None) -> None:
        MOAgent.__init__(self, env)
        MOPolicy.__init__(self, id)