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
    "vehicles_count": 18,
    "controlled_vehicles": 2,

    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        }
    }
}

run_config = {
    "env":  [env_config],

    "init": {
         "replay_buffer_size": [10_000],
         "batch_size" : [100],
         "reward_structure" : ["ego_reward", "mean_reward"],
         "use_multi_dqn" : [False,True],
         "observation_space_name" : ["Kinematics", "OccupancyGrid"],
    },
    "train": {
         "gamma": 0.99,
         "num_episodes" : 3000,
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
gridsearch(MOMA_DQN, env, run_config, seed=11, csv_file_path="data/moma_highway_test/", experiment_name="moma_highway")