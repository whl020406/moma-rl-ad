from circle_env import *
from mo_circle_env import *
# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from gymnasium.envs.registration import register
register(
        id='circle-env-v0',
        entry_point='envs:CircleEnv',
    )
register(
        id='mo-circle-env-v0',
        entry_point='envs:MOCircleEnv',
    )