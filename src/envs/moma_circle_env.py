# ✅ 文件：moma_circle_env.py（已修复）
from typing import Dict, Text, TypeVar
import numpy as np
from highway_env import utils
from highway_env.envs import AbstractEnv, RoadNetwork, Road, LineType, CircularLane
from src.energy_calculation import NaiveEnergyCalculation
from highway_env.vehicle.controller import ControlledVehicle
from circle_env import CircleEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.observation import Optional
from highway_env.envs.common.action import Action
from src.observations import AugmentedMultiAgentObservation
from highway_env.envs.common.action import action_factory
from highway_env.utils import near_split
import torch
from src.utils import random_objective_weights
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle

Observation = TypeVar("Observation")

class MOMACircleEnv(CircleEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                },
                "lateral": True
            },
            "screen_width": 501,
            "screen_height": 500,
            "lanes_count": 1,
            "inner_lane_radius": 80,
            "vehicles_count": 5,
            "controlled_vehicles": 2,
            "vehicles_density": 0.5,
            "duration": 100,
            "collision_reward": -1,
            "high_speed_reward": 1,
            "lane_change_reward": -0.05,
            "energy_consumption_reward": 1,
            "right_lane_reward": 0.1,
            "normalize_reward": True,
            "energy_consumption_function": NaiveEnergyCalculation,
            "max_speed": 20,
            "min_speed": 5,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "rng": np.random.default_rng(None),
            "ego_spacing": 2,
            "vehicles_density": 1,
        })
        config["action"]["action_config"]["target_speeds"] = np.linspace(
            config["min_speed"], config["max_speed"], endpoint=True, num=7
        )
        return config

    def _reward(self, action) -> np.ndarray:
        num_close_vehicles = self.observation_type.agents_observation_types[0].vehicles_count
        reward_dict_lists = self._rewards(action)
        reward_array = np.full(
            shape=(len(reward_dict_lists), num_close_vehicles, 3), fill_value=np.nan
        )
        for i, dict_list in enumerate(reward_dict_lists):
            for j, reward_dict in enumerate(dict_list):
                reward_array[i, j, :] = self._compute_vehicle_reward(reward_dict)
        return reward_array

    def _rewards(self, action) -> Dict[Text, float]:
        vehicle_lists = self.observation_type.curr_observation_vehicle_lists
        if not hasattr(self, 'energy_consumption_function'):
            self.energy_consumption_function = self.config["energy_consumption_function"](
                self.vehicle.target_speeds, self.vehicle.KP_A
            )

        reward_dict_lists = []
        for v_list in vehicle_lists:
            dict_list = []
            for vehicle in v_list:
                neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
                lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
                    else vehicle.lane_index[2]
                forward_speed = vehicle.speed * np.cos(vehicle.heading)
                scaled_speed = utils.lmap(forward_speed, [self.config["min_speed"], self.config["max_speed"]], [0, 1])

                dict = {
                    "collision_reward": float(vehicle.crashed),
                    "high_speed_reward": np.clip(scaled_speed, 0, 1),
                    "energy_consumption_reward": self.energy_consumption_function.compute_efficiency(
                        vehicle, normalise=self.config["normalize_reward"]
                    ),
                }
                dict_list.append(dict)
            reward_dict_lists.append(dict_list)
        return reward_dict_lists

    def _compute_vehicle_reward(self, reward_dict):
        rewards = reward_dict
        scalarised_rewards = {
            name: self.config.get(name, 0) * reward for name, reward in rewards.items()
        }
        speed_reward = scalarised_rewards["high_speed_reward"]
        energy_reward = scalarised_rewards["energy_consumption_reward"]
        safety_reward = 0.0
        if rewards["collision_reward"] != 0:
            safety_reward = self.config["collision_reward"]

        if self.config["normalize_reward"]:
            speed_reward = utils.lmap(speed_reward, [0, self.config["high_speed_reward"]], [0, 1])
            energy_reward = utils.lmap(energy_reward, [0, self.config["energy_consumption_reward"]], [0, 1])
            safety_reward = utils.lmap(safety_reward, [self.config["collision_reward"], 0], [0, 1])

        return np.array([speed_reward, energy_reward, safety_reward])

    def __normalize_rewards(self, rewards):
        speed_reward = rewards[0]
        energy_reward = rewards[1]
        safety_reward = rewards[2]

        speed_reward = utils.lmap(speed_reward, [0, self.config["high_speed_reward"]], [0, 1])
        energy_reward = utils.lmap(energy_reward, [0, self.config["energy_consumption_reward"]], [0, 1])
        safety_reward = utils.lmap(safety_reward, [self.config["collision_reward"], 0], [0, 1])

        return speed_reward, energy_reward, safety_reward

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        info = {
            "speed": self.vehicle.speed,
            "crashed": [v.crashed for v in self.controlled_vehicles],
            "vehicle_objective_weights": [
                [v.objective_weights for v in close_vehicles]
                for close_vehicles in self.observation_type.curr_observation_vehicle_lists
            ],
            "action": action,
        }
        return info

    def define_spaces(self) -> None:
        self.observation_type = AugmentedMultiAgentObservation(
            env=self, observation_config=self.unwrapped.config["observation"]
        )
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()


    def _make_vehicles(self) -> None:
        lane_length = self.road.network.graph[0][1][0].length * 3 # three implicit road sections, each accounting for 120 degrees

        num_vehicles = self.config["vehicles_count"]
        num_controlled_vehicles = self.config["controlled_vehicles"]
        total_vehicles = num_vehicles + num_controlled_vehicles
        vehicle_spacing = lane_length / total_vehicles

        placed_vehicles = 0

        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(num_vehicles, num_bins=num_controlled_vehicles)

        self.controlled_vehicles = []
        lane_counter = np.zeros(len(self.road.network.graph.keys()), dtype=int)
        for others in other_per_controlled:

            #controlled vehicle
            vehicle = MOMACircleEnv.create_at(
                 Vehicle,
                 self.road,
                 speed=None,
                 lane_from=0,
                 position=vehicle_spacing * placed_vehicles
            )
            placed_vehicles += 1

            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            vehicle.is_controlled = 1
            #set random objective weights for controlled vehicles (2-objectives)
            #can be overriden during training by the MOMA-RL-algorithm
            vehicle.objective_weights = random_objective_weights(num_objectives=2, rng = self.config["rng"], device= self.config["device"])
            
            #add controlled vehicle to list
            vehicle.MAX_SPEED = self.config["max_speed"]
            vehicle.MIN_SPEED = self.config["min_speed"]
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            #uncontrolled vehicles (non-autonomous)
            for _ in range(others):
                chosen_lane = np.argmin(lane_counter)
                lane_counter[chosen_lane] += 1

                vehicle = MOMACircleEnv.create_at(
                other_vehicles_type,
                self.road,
                speed=None,
                lane_from=0,
                position=vehicle_spacing * placed_vehicles
                )
                placed_vehicles += 1
                
                vehicle.randomize_behavior()
                vehicle.is_controlled = 0

                #set weights of 0.0 for each objective for uncontrolled vehicles (2-objectives)
                vehicle.MAX_SPEED = self.config["max_speed"]
                vehicle.MIN_SPEED = self.config["min_speed"]
                vehicle.objective_weights = torch.tensor([0.0,0.0], device=self.config["device"])
                self.road.vehicles.append(vehicle)

    def create_at(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      position: float = None) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        DISTANCE_WANTED_RANGE = [1,10]
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.5*lane.speed_limit, 0.9*lane.speed_limit)
            else:
                speed = road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        v = cls(road, lane.position(position, 0), lane.heading_at(position), speed)
        if cls == IDMVehicle:
            v.DISTANCE_WANTED = np.random.rand() * (DISTANCE_WANTED_RANGE[1] - DISTANCE_WANTED_RANGE[0]) + DISTANCE_WANTED_RANGE[0]
        return v
