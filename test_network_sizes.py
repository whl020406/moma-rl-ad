from src.MO_DQN import MO_DQN
import mo_gymnasium as mo_gym
from src.gridsearch import gridsearch
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

#experiment configurations
env_config_1 = {
        "collision_reward": -1,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "OccupancyGrid",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                "grid_step": [5, 5],
            }
        }
    }

env_config_2 = {
        "collision_reward": -1,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
            }
        }
    }

#kinematics with additional features
env_config_3 = {
        "collision_reward": -1,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "features": ['presence', 'x', 'y', 'vx', 'vy', "heading", "lat_off"],
            }
        }
    }

run_config = {
    "env": [env_config_1, env_config_2, env_config_3],
    "init": {
         "gamma": [0.9],
         "replay_buffer_size": [1000],
         "use_reward_normalisation_wrapper": [False],
         "use_default_reward_normalisation": [True],
         "network_hidden_sizes": [[128,128],[128,128,128],[256,256], [256,256,256]],

    },
    "train": {
         "num_iterations" : 100_000,
         "inv_target_update_frequency": 20,
         "epsilon_start": 0.9,
         "epsilon_end": 0,
         "epsilon_end_time": 0.7
    },
    "eval": {
        "num_repetitions": 20,
        "num_points": 30,
        "episode_recording_interval": None,
        "render_episodes": False
    },
}

#run the experiments
env = mo_gym.make('mo-highway-env-v0', render_mode='rgb_array')
gridsearch(MO_DQN, env, run_config, 11, csv_file_path="data/network_sizes_kinematics_with_lane/")