from typing import Dict, Text
import numpy as np
from highway_env import utils
from highway_env.envs import AbstractEnv, RoadNetwork, Road, LineType, CircularLane
from src.energy_calculation import NaiveEnergyCalculation
from highway_env.vehicle.controller import ControlledVehicle
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
            "right_lane_reward": 0.1,
            "normalize_reward": True,
            "energy_consumption_function": NaiveEnergyCalculation
            })
        return config

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        scalarised_rewards = {
            name: self.config.get(name, 0) * reward for name, reward in rewards.items()
        }
        #merge some rewards together (penalty rewards and right lane reward with high-speed and energy reward) corresponding to user-defined parameter function
        speed_reward =  scalarised_rewards["high_speed_reward"] + scalarised_rewards["lane_change_reward"] + scalarised_rewards["right_lane_reward"]
        energy_reward = scalarised_rewards["energy_consumption_reward"] + scalarised_rewards["lane_change_reward"] + scalarised_rewards["right_lane_reward"]
        
        if self.config["normalize_reward"]:
            speed_reward, energy_reward =  self.__normalize_rewards([speed_reward, energy_reward])

        #rewards["collision_reward"] indicates whether there has been a crash
        if rewards["collision_reward"] != 0:
            speed_reward = self.config["collision_reward"]
            energy_reward = self.config["collision_reward"]
            
        return np.array([speed_reward, energy_reward])
    
    def _rewards(self, action: int) -> Dict[Text, float]:
        #if its the first time this function is called: initialise energy consumption function
        if not hasattr(self, 'energy_consumption_function'):
            self.energy_consumption_function = self.config["energy_consumption_function"](self.vehicle.target_speeds, self.vehicle.KP_A)
        
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
            / (self.vehicle.target_speeds.size - 1), #this reward is always normalised
            "energy_consumption_reward": self.energy_consumption_function.compute_efficiency(self.vehicle, normalise=self.config["normalize_reward"]),
            "lane_change_reward": action in [0, 2],
            "right_lane_reward":  1 - (lane / max(len(neighbours) - 1, 1)), # 1 - so that the outer lane receives the highest reward

        }
    
    def __normalize_rewards(self, rewards):
        speed_reward = rewards[0]
        energy_reward = rewards[1]

        speed_reward = utils.lmap(speed_reward,
                                [self.config["lane_change_reward"],
                                    self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        
        energy_reward = utils.lmap(energy_reward,
                                [self.config["lane_change_reward"],
                                    self.config["energy_consumption_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        
        return speed_reward, energy_reward
        