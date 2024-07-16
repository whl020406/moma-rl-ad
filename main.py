import gymnasium as gym
import mo_gymnasium as mo_gym
from matplotlib import pyplot as plt
import numpy as np
from src import __init__ #initialises important packages / environments
from src.utils import AugmentedMultiAgentObservation
from mo_gymnasium import MONormalizeReward
from highway_env.road.lane import StraightLane
from highway_env.envs.common.observation import OccupancyGridObservation

env = gym.make('mo-highway-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "fps": 20,
    "screen_height": 500,
    "vehicles_count": 10,
    "controlled_vehicles": 1,
    "duration": 80,  # [s]

    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "OccupancyGrid",
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            #"grid_size": [[-27.5, 27.5], [StraightLane.DEFAULT_WIDTH * (-2), StraightLane.DEFAULT_WIDTH * 2]],
            "grid_size": [[-10, 10], [-20,20]],
            "grid_step": [5, 5],
            "align_to_vehicle_axes": False,
        }
    },
})
env.unwrapped.configure({
    "manual_control": True
})
env.reset()
done = False
truncated = False
while True:
    #print(env.get_wrapper_attr('get_available_actions')())
    obs, reward, done, truncated, info = env.step(1)
    #print(env.unwrapped.controlled_vehicles[0].position)
    print(obs[0][0])
    env.render()

plt.imshow(env.render())
plt.show()