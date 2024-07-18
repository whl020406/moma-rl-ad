import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from src import MO_DQN
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd

env = mo_gym.make('mo-highway-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "screen_height": 500,
    "lanes_count": 4,
    "vehicles_count": 20,
    #"collision_reward": -1,    # The reward received when colliding with a vehicle.
    #"right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
    #"high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
    #"lane_change_reward": 0,   # The reward received at each lane change action.
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "normalize": True,
        }
    }
})
env.unwrapped.configure({
    "manual_control": False
})
import torch.nn as nn
obs, info = env.reset()
agent = MO_DQN.MO_DQN(env, loss_criterion=nn.SmoothL1Loss, num_objectives=2, seed=11, observation_space_shape=obs[0].shape, replay_buffer_size=100, batch_ratio=0.6, objective_names=["speed_reward", "energy_reward"])
#df = agent.train(1000, epsilon_start=0.9, epsilon_end=0.05, inv_optimisation_frequency=1, num_evaluations=2)
agent.evaluate()
#df.plot.line(x="iteration", y="loss")
#plt.show()