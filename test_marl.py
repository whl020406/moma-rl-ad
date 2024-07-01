import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from src import MOMA_DQN
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd

env = mo_gym.make('moma-highway-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "screen_height": 500,
    "vehicles_count": 10,
    "controlled_vehicles": 2,
    "observation": {
        "type": "AugmentedMultiAgentObservation",
        "observation_config": {
            "type": "AugmentedMultiAgentObservation",
            "see_behind": False
        }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        }
    }
})

obs, info = env.reset()

moma_agent = MOMA_DQN.MOMA_DQN(env, num_objectives=2, seed=11, observation_space_shape=obs[0].shape, replay_buffer_size=200, batch_ratio=0.3,
                      objective_names=["speed_reward", "energy_reward"])
moma_agent.train(50_000, epsilon_start=0.9, epsilon_end=0.0)

df = moma_agent.evaluate(num_repetitions= 20, hv_reference_point=np.ndarray([0,0]), seed=11, episode_recording_interval=None)
print(df)
df.to_csv("data/linear_scalarisation_eval.csv")