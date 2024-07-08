import gymnasium as gym
import mo_gymnasium as mo_gym
from matplotlib import pyplot as plt
import numpy as np
from src import __init__ #initialises important packages / environments
from src.utils import AugmentedMultiAgentObservation
from mo_gymnasium import MONormalizeReward
env = gym.make('mo-highway-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "fps": 20,
    "screen_height": 500,
    "vehicles_count": 2,
    "controlled_vehicles": 1,
    "duration": 80,  # [s]

    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "see_behind": False,
            "normalize": False,
            "features": ['presence', 'x', 'y', 'vx', 'vy']
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
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    print(obs[0])
    env.render()

plt.imshow(env.render())
plt.show()