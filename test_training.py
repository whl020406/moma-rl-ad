import gymnasium as gym
import mo_gymnasium as mo_gym
from matplotlib import pyplot as plt
import numpy as np
from src import MO_DQN

env = mo_gym.make('circle-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "screen_height": 500,
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
        }
    }
})
env.unwrapped.configure({
    "manual_control": True
})

obs, info = env.reset()

agent = MO_DQN.MO_DQN(env, num_objectives=1, seed=11, observation_space_shape=obs[0].shape)
agent.train()