import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from src import MOMA_DQN
from src import MO_DQN
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import torch

env = mo_gym.make('moma-highway-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
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
})
env.unwrapped.configure({
    "manual_control": False
})

obs, info = env.reset()
obs = [torch.tensor(single_obs) for single_obs in obs] #reshape observations and
obs = [single_obs[~torch.isnan(single_obs)].reshape(1,-1) for single_obs in obs] #remove nan values
agent = MOMA_DQN.MOMA_DQN(env, num_objectives=2, seed=11, replay_buffer_size=1000, batch_ratio=0.3, use_multi_dqn=False, reward_structure="mean_reward", observation_space_name="Kinematics")
loss_logger_df = agent.train(10_000, epsilon_start=0.9, epsilon_end=0, epsilon_end_time= 0.8, num_evaluations=0)
eval_logger_df, vehicle_logger_df = agent.evaluate(seed=11, episode_recording_interval=10, video_name_prefix="Test_MOMA_3")

eval_logger_df.to_csv("test_4_kinematics_256_laneinfos/eval_test.csv")
loss_logger_df.to_csv("test_4_kinematics_256_laneinfos/loss_test.csv")
vehicle_logger_df.to_csv("test_4_kinematics_256_laneinfos/vehicle_test.csv")

agent.store_network("test_4_kinematics_256_laneinfos/", "model_test.pth")