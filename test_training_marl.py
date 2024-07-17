import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from src import MOMA_DQN
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import torch

env = mo_gym.make('moma-highway-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "screen_height": 500,
    "lanes_count": 4,
    "vehicles_count": 10,
    "controlled_vehicles": 2,
    "observation": {
        "type": "AugmentedMultiAgentObservation",
        "observation_config": {
            "type": "AugmentedMultiAgentObservation",
            "see_behind": False
        }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        }
    }
})
env.unwrapped.configure({
    "manual_control": False
})

obs, info = env.reset()
obs = [torch.tensor(single_obs) for single_obs in obs] #reshape observations and
obs = [single_obs[~torch.isnan(single_obs)].reshape(1,-1) for single_obs in obs] #remove nan values

agent = MOMA_DQN.MOMA_DQN(env, num_objectives=2, seed=11, observation_space_length=obs[0].shape[1], replay_buffer_size=100, batch_ratio=0.6, objective_names=["speed_reward", "energy_reward"])
df = agent.train(500, epsilon_start=0.9, epsilon_end=0.05, inv_optimisation_frequency=1, num_evaluations=2)
print(df)