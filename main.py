import gymnasium as gym
import mo_gymnasium as mo_gym
from matplotlib import pyplot as plt
import numpy as np

env = mo_gym.make('circle-env-v0', render_mode='rgb_array')
env.unwrapped.configure({
    "screen_width": 500,
    "screen_height": 500,
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
        }
    }
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
    print(info)
    env.render()
    #print(obs)

plt.imshow(env.render())
plt.show()