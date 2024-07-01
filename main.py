import gymnasium as gym
import mo_gymnasium as mo_gym
from matplotlib import pyplot as plt
import numpy as np
from src import __init__ #initialises important packages / environments
from src.utils import AugmentedMultiAgentObservation

env = gym.make('mo-highway-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "fps": 20,
    "screen_height": 500,
    "vehicles_count": 2,
    "controlled_vehicles": 2,
    "duration": 80,  # [s]

    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "see_behind": True,
            "normalize": True
        }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        }
    }
})
env.unwrapped.configure({
    "manual_control": True
})
env.reset()
env.unwrapped.observation_type = AugmentedMultiAgentObservation(env = env, **env.unwrapped.config["observation"])
env.unwrapped.observation_space = env.observation_type.space()
done = False
truncated = False
while True:
    #print(env.get_wrapper_attr('get_available_actions')())
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(obs)
    env.render()

plt.imshow(env.render())
plt.show()