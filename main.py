import gymnasium as gym
import mo_gymnasium as mo_gym
from matplotlib import pyplot as plt
import numpy as np
from src import __init__ #initialises important packages / environments
from src.observations import AugmentedMultiAgentObservation
from mo_gymnasium import MONormalizeReward
from highway_env.road.lane import StraightLane
from highway_env.envs.common.observation import OccupancyGridObservation

env = gym.make('moma-circle-env-v0', render_mode='rgb_array')
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
            "features": ["presence", "x", "y", "vx", "vy", "on_road"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "grid_size": [[-27.5, 27.5], [-27.5, +27.5]],
            "grid_step": [5, 5],
            "align_to_vehicle_axes": False,
        }
    },
})
env.unwrapped.configure({
    "manual_control": True
})
env.reset()
env.unwrapped.controlled_vehicles[0].check_collisions = False
env.unwrapped.controlled_vehicles[0].collidable = False
done = False
truncated = False
while True:
    obs, reward, done, truncated, info = env.step(1)
    print(obs[0][5])
    env.render()

plt.imshow(env.render())
plt.show()