import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from src import MOMA_DQN
from src import MO_DQN
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import torch

env = mo_gym.make('moma-circle-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "screen_height": 500,
    "vehicles_count": 20,
    "num_lanes": 1,
    "inner_lane_radius": 60,
    "controlled_vehicles": 1,
    "max_speed" : 10,
    "min_speed" : 5,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        }
    }
})
 
env.unwrapped.configure({
    "manual_control": True
})
env.reset()
env.unwrapped.controlled_vehicles[0].check_collisions = False
env.unwrapped.controlled_vehicles[0].collidable = False
done = False
truncated = False
while True:
    env.render()
    obs, reward, done, truncated, info = env.step(1)
    #print(env.unwrapped.controlled_vehicles[0].lane_index[0])