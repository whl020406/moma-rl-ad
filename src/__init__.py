import sys, os
print(os.path.join(os.path.dirname(__file__), "envs"))
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from envs.circle_env import *