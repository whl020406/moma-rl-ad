from src.MOMA_DQN import MOMA_DQN
import mo_gymnasium as mo_gym
from src.gridsearch import gridsearch
from tqdm.notebook import tqdm
import itertools
import pandas as pd
import numpy as np

#final environment config used for experiments
env_config_1 = {
    "screen_width": 700,
    "screen_height": 400,
    "vehicles_count": 18,
    "controlled_vehicles": 2,
}

env_config_2 = {
    "screen_width": 700,
    "screen_height": 400,
    "vehicles_count": 12,
    "controlled_vehicles": 8,
}

env_config_3 = {
    "screen_width": 700,
    "screen_height": 400,
    "vehicles_count": 6,
    "controlled_vehicles": 14,
}

run_config = {
    "env":  [env_config_1, env_config_2, env_config_3],

    "init": {
         "replay_buffer_size": [10_000],
         "batch_size" : [100],
         "reward_structure" : ["ego_reward", "mean_reward"],
         "use_multi_dqn" : [False,True],
         "observation_space_name" : ["OccupancyGrid"],
         "increase_ego_reward_importance": [True],
         "estimate_uncontrolled_obj_weights": [True],
    },
    "train": {
         "gamma": 0.99,
         "num_episodes" : 2000,
         "inv_target_update_frequency": 10,
         "epsilon_start": 0.9,
         "epsilon_end": 0.1,
         "epsilon_end_time": 0.75,
         "num_evaluations": 10,
    },
    "eval": {
        "num_repetitions": 20,
        "num_points": 20,
    },
}

env = mo_gym.make('moma-highway-env-v0', render_mode='rgb_array')
gridsearch(MOMA_DQN, env, run_config, seed=11, csv_file_path="data/moma_highway_test_final_estimates_weights/", experiment_name="moma_highway_test_final_estimates_weights", only_evaluate=True)