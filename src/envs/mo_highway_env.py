from typing import Dict, Text
import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from utils import calc_energy_efficiency, compute_max_energy_consumption

class MOHighwayEnv(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "energy_consumption_reward": 1,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reward(self, action: Action) -> float:
        
        rewards = self._rewards(action)
        rewards = {
            name: self.config.get(name, 0) * reward for name, reward in rewards.items()
        }
        speed_reward = rewards["high_speed_reward"] + rewards["collision_reward"] + rewards["right_lane_reward"]
        energy_reward = rewards["energy_consumption_reward"] + rewards["collision_reward"] + rewards["right_lane_reward"]
        if self.config["normalize_reward"]:
            speed_reward = utils.lmap(speed_reward,
                                [self.config["collision_reward"],
                                    self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
            energy_reward = utils.lmap(energy_reward,
                                [self.config["collision_reward"],
                                    self.config["energy_consumption_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        return [speed_reward, energy_reward]

    def _rewards(self, action: Action) -> Dict[Text, float]:
        #if its the first time this function is called: calculate maximum energy consumption:
        if not hasattr(self, 'max_energy_consumption'):
            self.max_energy_consumption = compute_max_energy_consumption(self.vehicle)

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "energy_consumption_reward": calc_energy_efficiency(self.vehicle, normalise=self.config["normalize_reward"], max_energy_consumption=self.max_energy_consumption)
        }