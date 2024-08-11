from src.MOMA_DQN import MOMA_DQN
import mo_gymnasium as mo_gym
from src.gridsearch import gridsearch
from tqdm.notebook import tqdm
import itertools
import pandas as pd
import numpy as np

#final environment config used for experiments
env_config_1 = {
    "screen_width": 501,
    "screen_height": 500,
    "vehicles_count": 12,
    "controlled_vehicles": 2,
    "max_speed" : 20,
    "min_speed" : 5,
}

env_config_2 = {
    "screen_width": 501,
    "screen_height": 500,
    "vehicles_count": 7,
    "controlled_vehicles": 7,
    "max_speed" : 20,
    "min_speed" : 5,
}

env_config_3 = {
    "screen_width": 501,
    "screen_height": 500,
    "vehicles_count": 4,
    "controlled_vehicles": 10,
    "max_speed" : 20,
    "min_speed" : 5,
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

env = mo_gym.make('moma-circle-env-v0', render_mode='rgb_array')
gridsearch(MOMA_DQN, env, run_config, seed=11, csv_file_path="data/moma_circle_final/", experiment_name="moma_circle_final")