from src.MOMA_DQN import MOMA_DQN
import mo_gymnasium as mo_gym
from src.gridsearch import gridsearch
from tqdm.notebook import tqdm
import itertools
import pandas as pd
import numpy as np

#final environment config used for experiments
env_config = {
    "screen_width": 500,
    "screen_height": 500,
    "vehicles_count": 10,
    "controlled_vehicles": 2,

    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        }
    }
}

# run_config = {
#     "env":  [env_config],

#     "init": {
#          "replay_buffer_size": [1000],
#          "batch_ratio" : [0.2],
#          "reward_structure" : ["ego_reward", "mean_reward"],
#          "observation_space_name" : ["Kinematics", "OccupancyGrid"],
#     },
#     "train": {
#          "gamma": 0.9,
#          "num_episodes" : 25_000,
#          "inv_target_update_frequency": 10,
#          "epsilon_start": 0.9,
#          "epsilon_end": 0,
#          "epsilon_end_time": 0.75,
#     },
#     "eval": {
#         "num_repetitions": 20,
#         "num_points": 20,
#         "num_evaluations": 10,
#     },
# }

#test run config for generating example data
run_config = {
    "env":  [env_config],

    "init": {
         "replay_buffer_size": [1000],
         "batch_ratio" : [0.2],
         "reward_structure" : ["ego_reward", "mean_reward"],
         "observation_space_name" : ["Kinematics", "OccupancyGrid"],
    },
    "train": {
         "gamma": 0.9,
         "num_episodes" : 100,
         "inv_target_update_frequency": 10,
         "epsilon_start": 0.9,
         "epsilon_end": 0,
         "epsilon_end_time": 0.75,
         "num_evaluations": 2
    },
    "eval": {
        "num_repetitions": 2,
        "num_points": 3,
    },
}


env = mo_gym.make('moma-highway-env-v0', render_mode='rgb_array')
gridsearch(MOMA_DQN, env, run_config, 11, csv_file_path="data/moma_highway_test/", experiment_name="moma_highway")