# ✅ 文件：envs/__init__.py（统一注册）
from .circle_env import CircleEnv
from .mo_circle_env import MOCircleEnv
from .moma_circle_env import MOMACircleEnv
from .moma_highway_env import MOMAHighwayEnv

import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="moma-highway-env-v0",
    entry_point="envs.moma_highway_env:MOMAHighwayEnv",
)

register(
    id="moma-circle-env-v0",
    entry_point="envs.moma_circle_env:MOMACircleEnv",
)