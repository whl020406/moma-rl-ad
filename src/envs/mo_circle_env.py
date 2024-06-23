from typing import Dict, Text
import numpy as np
from highway_env import utils
from highway_env.envs import AbstractEnv, RoadNetwork, Road, LineType, CircularLane
from utils import calc_energy_efficiency, compute_max_energy_consumption
from circle_env import CircleEnv
from highway_env.vehicle.controller import MDPVehicle

class MOCircleEnv(CircleEnv):
        
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
            "collision_reward": -1,
            "high_speed_reward": 1,
            "lane_change_reward": -0.05,
            "energy_consumption_reward": 1,
            "normalize_reward": True,
            #add energy reward
            #add lane reward
            })
        return config

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        rewards = {
            name: self.config.get(name, 0) * reward for name, reward in rewards.items()
        }
        #merge some rewards together (penalty rewards and right lane reward with high-speed and energy reward) corresponding to user-defined parameter function
        rewards = np.array([
            rewards["high_speed_reward"] + rewards["collision_reward"] + rewards["lane_change_reward"],
            rewards["energy_consumption_reward"] + rewards["collision_reward"] + rewards["lane_change_reward"]
        ])
        return rewards

    def _rewards(self, action: int) -> Dict[Text, float]:
        #if its the first time this function is called: calculate maximum energy consumption:
        if not hasattr(self, 'max_energy_consumption'):
            self.max_energy_consumption = compute_max_energy_consumption(self.vehicle)
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
            / (self.vehicle.target_speeds.size - 1), #this reward is always normalised
            "energy_consumption_reward": calc_energy_efficiency(self.vehicle, normalise=self.config["normalize_reward"], max_energy_consumption=self.max_energy_consumption),
            "lane_change_reward": action in [0, 2],
        }
        