from src.MO_DQN import MO_DQN
import mo_gymnasium as mo_gym
from src.gridsearch import gridsearch
from tqdm.notebook import tqdm
import itertools
import pandas as pd
import numpy as np

obs_space_1 = {
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
obs_space_2 ={
            "type": "AugmentedMultiAgentObservation",
            "observation_config": {
                "type": "Kinematics"
            }
        }
obs_space_3 = {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics"
            }
          }

run_config = {
    "env": {
        "observation": [obs_space_1, obs_space_2, obs_space_3]
    },
    "init": {
         "gamma": [0.9],
         "replay_buffer_size": [200, 1000],
         "use_reward_normalisation_wrapper": [False, True]
    },
    "train": {
         "num_iterations" : 500, #change to 200_000
         "inv_target_update_frequency": 20,
         "epsilon_start": 0.9,
         "epsilon_end": 0
    },
    "eval": {
        "num_repetitions": 20,
        "num_points": 30,
        "hv_reference_point": np.ndarray([0,0]),
        "episode_recording_interval": None,
        "render_episodes": False
    },
}


env = mo_gym.make('mo-highway-env-v0', render_mode='rgb_array')
gridsearch(MO_DQN, env, run_config, 11)