from typing import Dict, Text
import numpy as np
from highway_env import utils
from highway_env.envs import AbstractEnv, RoadNetwork, Road, LineType, CircularLane
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

class CircleEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 100,
            "normalize_reward": True,
            "fixed_centering_position":np.array([0,10])
            })
        config["inner_lane_radius"] = 40
        config["num_lanes"] = 3
        config["vehicles_count"] = 6
        config["controlled_vehicles"] = 1
        config["vehicles_density"] = 0.1
        config["max_speed"] = 10
        config["min_speed"] = 2
        config["lane_width"] = 6

        config["normalize_reward"] = True
        config["action"] = {"type": "DiscreteMetaAction", "target_speeds": np.linspace(config["min_speed"], config["max_speed"], endpoint=True, num=5), "lateral":True}
        return config

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["high_speed_reward"]],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
            / (self.vehicle.target_speeds.size - 1),
            "lane_change_reward": action in [0, 2],
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self):
        '''
        This function constructs the road network.
        '''
        lane_width = self.config["lane_width"]
        self.inner_lane_radius = self.config["inner_lane_radius"]
        self.num_lanes = self.config["num_lanes"]
        network = RoadNetwork()
        radii = np.arange(start=self.inner_lane_radius, step=lane_width, stop=self.inner_lane_radius+lane_width*self.num_lanes)[::-1] #get radius for every lane
        #set line striping
        lane_stripes = [[LineType.STRIPED, LineType.NONE] for _ in range(self.num_lanes)] #every line is striped
        lane_stripes[0][0] = LineType.CONTINUOUS #except the outer lane
        lane_stripes[-1][1] = LineType.CONTINUOUS #and the inner lane

        for lane_id in range(self.num_lanes):

            network.add_lane("a","b", CircularLane(
                [0,0], radii[lane_id], start_phase=np.deg2rad(0), end_phase=np.deg2rad(179), line_types=lane_stripes[lane_id], speed_limit=self.config["max_speed"], width=lane_width
            ))
            
            network.add_lane("b","a", CircularLane(
                [0,0], radii[lane_id], start_phase=np.deg2rad(180), end_phase=np.deg2rad(359), line_types=lane_stripes[lane_id], speed_limit=self.config["max_speed"], width=lane_width
            ))

        road = Road(network, np_random= self.np_random, record_history=self.config["show_trajectories"])
        self.road = road
    
    def _make_vehicles(self):
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """

        # Other vehicles
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            #ego vehicle
            vehicle = Vehicle.create_random(
                self.road,
                speed=0.5*(self.config["max_speed"]-self.config["min_speed"])+self.config["min_speed"],
                lane_id=self.np_random.integers(0, self.num_lanes),
                spacing=1,
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            vehicle.MAX_SPEED = self.config["max_speed"]
            vehicle.MIN_SPEED = self.config["min_speed"]
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            #other vehicles
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.np_random.integers(0, self.num_lanes),
                    speed=0.5*(self.config["max_speed"]-self.config["min_speed"])+self.config["min_speed"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
        