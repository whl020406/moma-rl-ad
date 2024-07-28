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
    "screen_width": 700,
    "screen_height": 400,
    "vehicles_count": 10,
    "controlled_vehicles": 2,
    "observation": {
        "type": "AugmentedMultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "see_behind": False,
        }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        }
    }
})

obs, info = env.reset()
obs = [torch.tensor(single_obs) for single_obs in obs] #reshape observations and
obs = [single_obs[~torch.isnan(single_obs)].reshape(1,-1) for single_obs in obs] #remove nan values
moma_agent = MOMA_DQN.MOMA_DQN(env, num_objectives=2, seed=11, replay_buffer_size=10, batch_ratio=0.3,
                      objective_names=["speed_reward", "energy_reward"], observation_space_name="OccupancyGrid", reward_structure="mean_reward")
moma_agent.train(1000, epsilon_start=0.9, epsilon_end=0.0, epsilon_end_time=0.7)
input("Press Enter to continue...")
moma_agent.evaluate(render_episodes=True)
#df = moma_agent.evaluate(num_repetitions= 2, num_points=3, seed=11, episode_recording_interval=None)
#print(df)