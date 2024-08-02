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
    "screen_width": 700,
    "screen_height": 300,
    "vehicles_count": 10,
    "controlled_vehicles": 2,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        }
    }
})

agent = MOMA_DQN.MOMA_DQN(env, num_objectives=2, seed=11, replay_buffer_size=10_000, batch_size=100, use_multi_dqn=False, reward_structure="mean_reward", observation_space_name="OccupancyGrid")
loss_logger_df = agent.train(1000, epsilon_start=0.9, epsilon_end=0.1, epsilon_end_time= 0.8, num_evaluations=2)
eval_logger_df, vehicle_logger_df = agent.evaluate(seed=11, video_name_prefix="Test_MOMA_7", render_episodes=False, hv_reference_point=np.ndarray([0,0]))

eval_logger_df.to_csv("test_7_mean_reward/eval_test.csv")
loss_logger_df.to_csv("test_7_mean_reward/loss_test.csv")
vehicle_logger_df.to_csv("test_7_mean_reward/vehicle_test.csv")

agent.store_network("test_7_mean_reward/", "model_test.pth")