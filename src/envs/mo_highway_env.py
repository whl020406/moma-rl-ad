from typing import Dict, Text
import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import near_split
from src.energy_calculation import NaiveEnergyCalculation
import torch
from src.utils import random_objective_weights
from highway_env.envs.common.action import action_factory, Action
from highway_env.envs.common.observation import observation_factory
from src.observations import AugmentedMultiAgentObservation

class MOHighwayEnv(HighwayEnvFast):
    '''Extends the standard highway environment to work with multiple objectives. The code was taken straight
    from the HighwayEnv class of the highway_env module and adjusted at various points.'''

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "screen_width": 800,
            "screen_height": 500,
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 80,  # [s]
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
            "offroad_terminal": False,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), #uses GPU if possible
            "energy_consumption_function": NaiveEnergyCalculation,
            "rng": np.random.default_rng(None) #sets random seed for rng by default
        })
        return config

    def _reward(self, action: Action) -> float:

        rewards = self._rewards(action)
        scalarised_rewards = {
            name: self.config.get(name, 0) * reward for name, reward in rewards.items()
        }
        speed_reward = scalarised_rewards["high_speed_reward"] + scalarised_rewards["right_lane_reward"]
        energy_reward = scalarised_rewards["energy_consumption_reward"] + scalarised_rewards["right_lane_reward"]

        #default normalisation
        if self.config["normalize_reward"]:
            speed_reward, energy_reward = self.__normalize_rewards([speed_reward, energy_reward])
        else:
            self.raw_rewards = scalarised_rewards #store raw rewards for info function
        
        #rewards["collision_reward"] indicates whether there has been a crash
        if rewards["collision_reward"] != 0:
           speed_reward = self.config["collision_reward"]
           energy_reward = self.config["collision_reward"]
                   
        return np.array([speed_reward, energy_reward])

    def _rewards(self, action: Action) -> Dict[Text, float]:
        #if its the first time this function is called: initialise energy consumption function
        if not hasattr(self, 'energy_consumption_function'):
            self.energy_consumption_function = self.config["energy_consumption_function"](self.vehicle.target_speeds, self.vehicle.KP_A)

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
            "energy_consumption_reward": self.energy_consumption_function.compute_efficiency(self.vehicle, True)
        }
    
    def __normalize_rewards(self, rewards):
        speed_reward = rewards[0]
        energy_reward = rewards[1]

        speed_reward = utils.lmap(speed_reward,
                                [0,
                                    self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        
        energy_reward = utils.lmap(energy_reward,
                                [0,
                                    self.config["energy_consumption_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        
        return speed_reward, energy_reward

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        
        for others in other_per_controlled:
            #controlled vehicle
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            vehicle.is_controlled = 1
            #set random objective weights for controlled vehicles (2-objectives)
            #can be overriden during training by the MOMA-RL-algorithm
            vehicle.objective_weights = random_objective_weights(num_objectives=2, rng = self.config["rng"], device= self.config["device"])
            
            #add controlled vehicle to list
            max_speed = vehicle.target_speeds[-1]
            min_speed = vehicle.target_speeds[0]
            vehicle.MAX_SPEED = max_speed
            vehicle.MIN_SPEED = min_speed
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            #uncontrolled vehicles (non-autonomous)
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                vehicle.is_controlled = 0

                #set weights of 0.0 for each objective for uncontrolled vehicles (2-objectives)
                vehicle.MAX_SPEED = max_speed
                vehicle.MIN_SPEED = min_speed
                vehicle.objective_weights = torch.tensor([0.0,0.0], device=self.config["device"])
                self.road.vehicles.append(vehicle)
    
    def _info(self, obs, action = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["rewards"] = self._reward(action)
            #if not normalised, report normalised rewards in info dict
            if not self.config["normalize_reward"]:
                info["rewards"] = self.__normalize_rewards(info["rewards"])
        except NotImplementedError:
            pass
        return info

    def define_spaces(self) -> None:
        """
        Override this function originally defined in the AbstractEnv class to work with my augmented observation space
        """
        try:
            self.observation_type = observation_factory(self, self.config["observation"])
            self.action_type = action_factory(self, self.config["action"])
            self.observation_space = self.observation_type.space()
            self.action_space = self.action_type.space()
        except:
            if self.unwrapped.config["observation"]["type"] == "AugmentedMultiAgentObservation":
                self.observation_type = AugmentedMultiAgentObservation(env = self, observation_config=self.unwrapped.config["observation"])
                self.action_type = action_factory(self, self.config["action"])
                self.observation_space = self.observation_type.space()
                self.action_space = self.action_type.space()
            else:
                raise Exception("Observation space cannot be applied!")