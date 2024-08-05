from typing import Dict, Text, TypeVar
import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import near_split
from highway_env.envs.common.observation import *
from highway_env.envs.common.action import action_factory
from observations import AugmentedMultiAgentObservation
from energy_calculation import NaiveEnergyCalculation
import torch
from utils import random_objective_weights

Observation = TypeVar("Observation")

class MOMAHighwayEnv(HighwayEnvFast):
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
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                }
            },
            "lanes_count": 4,
            "vehicles_count": 10,
            "controlled_vehicles": 2,
            "initial_lane_id": None,
            "duration": 80,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.2,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "energy_consumption_reward": 1,
            "reward_speed_range": [15, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), #uses GPU if possible
            "energy_consumption_function": NaiveEnergyCalculation,
            "rng": np.random.default_rng(None), #sets random seed for rng by default
            "set_uncontrolled_obj_weights": False
        })
        config["action"]["action_config"]["target_speeds"] = np.linspace(config["reward_speed_range"][0], config["reward_speed_range"][1], endpoint=True, num=7)
        return config

    def _reward(self, action: Action) -> float:
        '''Fetches the reward dictionaries for all ego vehicles and their closest neighbouring cars and uses that 
           to construct a reward array, containing the two-dimensional rewards for each vehicle. 
           The first vehicle in the second dimension corresponds to an ego-vehicle'''
        
        num_close_vehicles = self.observation_type.agents_observation_types[0].vehicles_count
        reward_dict_lists = self._rewards(action)
        #rows with np.nan indicate missing close vehicle
        reward_array = np.full(shape=(len(reward_dict_lists),num_close_vehicles,2), fill_value=np.nan) #2 because we have two objectives
        
        for i, dict_list in enumerate(reward_dict_lists):
            for j, reward_dict in enumerate(dict_list):
                reward_array[i,j,:] = self._compute_vehicle_reward(reward_dict)
        return reward_array
    
    def _compute_vehicle_reward(self, reward_dict):
        '''Computes the reward tuple for a single vehicle based on the information in the reward dictionary.'''
        rewards = reward_dict
        scalarised_rewards = {
            name: self.config.get(name, 0) * reward for name, reward in rewards.items()
        }
        speed_reward = scalarised_rewards["high_speed_reward"] + scalarised_rewards["right_lane_reward"]
        energy_reward = scalarised_rewards["energy_consumption_reward"] + scalarised_rewards["right_lane_reward"]
        
        if self.config["normalize_reward"]:
            speed_reward, energy_reward = self.__normalize_rewards([speed_reward, energy_reward])

        #rewards["collision_reward"] indicates whether there has been a crash
        if rewards["collision_reward"] != 0:
           speed_reward = self.config["collision_reward"]
           energy_reward = self.config["collision_reward"]

        return np.array([speed_reward, energy_reward])
    
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

    def _rewards(self, action: Action) -> Dict[Text, float]:
        '''constructs the reward dictionaries for each vehicle using the variable curr_observation_vehicle_lists in
           AugmentedMultiAgentObservation.'''
        
        #fetch vehicles of the current observation from observation type
        vehicle_lists = self.observation_type.curr_observation_vehicle_lists
        #if its the first time this function is called: initialise energy consumption function
        if not hasattr(self, 'energy_consumption_function'):
            self.energy_consumption_function = self.config["energy_consumption_function"](self.vehicle.target_speeds, self.vehicle.KP_A)

        reward_dict_lists = [] #list containing a list of reward dicts for each vehicle
        for v_list in vehicle_lists:
            dict_list = [] #list containing the reward dicts for an ego-vehicle and it's close vehicles
            for vehicle in v_list:

                neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
                lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
                    else vehicle.lane_index[2]
                # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
                forward_speed = vehicle.speed * np.cos(vehicle.heading)
                scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

                dict = {
                    "collision_reward": float(vehicle.crashed),
                    "right_lane_reward": lane / max(len(neighbours) - 1, 1),
                    "high_speed_reward": np.clip(scaled_speed, 0, 1),
                    "energy_consumption_reward": self.energy_consumption_function.compute_efficiency(vehicle, normalise=self.config["normalize_reward"])
                }
                dict_list.append(dict)
            reward_dict_lists.append(dict_list)
        
        return reward_dict_lists

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        
        for others in other_per_controlled:
            #controlled vehicle
            vehicle = Vehicle.create_random(
                self.road,
                speed=None,
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
                if self.config["set_uncontrolled_obj_weights"]:
                    self.set_uncontrolled_vehicle_obj_weights(vehicle, min_speed, max_speed)
                else:
                    vehicle.objective_weights = torch.tensor([0.0,0.0], device=self.config["device"])
                self.road.vehicles.append(vehicle)

    def set_uncontrolled_vehicle_obj_weights(self, vehicle, min_speed, max_speed):
        target_speed = vehicle.target_speed
        speed_obj_weight = (target_speed - min_speed) / (max_speed - min_speed)
        vehicle.objective_weights = torch.tensor([speed_obj_weight, 1 - speed_obj_weight], device=self.config["device"])

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information. "crashed" features information on crashes for all controllable vehicles

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": [v.crashed for v in self.controlled_vehicles],
            "vehicle_objective_weights" : [[v.objective_weights for v in close_vehicles] 
                                           for close_vehicles in self.observation_type.curr_observation_vehicle_lists],
            "action": action,
        }
        return info

    def define_spaces(self) -> None:
        """
        Override this function originally defined in the AbstractEnv class to work with my augmented observation space
        """
        self.observation_type = AugmentedMultiAgentObservation(env = self, observation_config=self.unwrapped.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()
