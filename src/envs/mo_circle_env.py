from typing import Dict, Text
import numpy as np
from highway_env import utils
from highway_env.envs import AbstractEnv, RoadNetwork, Road, LineType, CircularLane
from circle_env import CircleEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

class MOCircleEnv(CircleEnv):
    
    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        rewards = np.array([
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        ])
        #returns reward vector
        return rewards

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
            / (self.vehicle.target_speeds.size - 1),
            #add another element regarding the energy consumption /CO2 emission
            "lane_change_reward": action in [0, 2],
            "on_road_reward": self.vehicle.on_road,
        }
        