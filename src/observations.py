import highway_env.utils
import torch
import numpy as np
from typing import List, Dict, TYPE_CHECKING, Tuple
import pandas as pd
from highway_env.envs.common.observation import KinematicObservation, OccupancyGridObservation, ObservationType
from gymnasium import spaces
from highway_env.vehicle.kinematics import Vehicle, Optional
from highway_env import utils
from highway_env.road.lane import AbstractLane   # ✅ 修正这里

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class AugmentedMultiAgentObservation(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []

        for vehicle in self.env.controlled_vehicles:
            if "observation_config" in observation_config:
                match observation_config["observation_config"]["type"]:
                    case "Kinematics":
                        obs_type = AugmentedKinematicObservation(env, **observation_config["observation_config"])
                    case "OccupancyGrid":
                        obs_type = AugmentedOccupancyGridObservation(env, **observation_config["observation_config"])
                    case _:
                        obs_type = AugmentedKinematicObservation(env, **observation_config["observation_config"])
            else:
                obs_type = AugmentedKinematicObservation(env, **observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])

    def observe(self) -> tuple:
        observations = []
        vehicle_lists = []
        for obs_type in self.agents_observation_types:
            obs, vehicle_list = obs_type.observe()
            observations.append(obs)
            vehicle_lists.append(vehicle_list)

        self.curr_observation_vehicle_lists = vehicle_lists
        return tuple(observations)


class AugmentedKinematicObservation(KinematicObservation):
    FEATURES: List[str] = [
        'presence', 'x', 'y', 'vx', 'vy',
        "heading", "lat_off", "long_off", "lane_info"
    ]

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = False,
                 observe_intentions: bool = False,
                 num_objectives: int = 3,  # 三目标
                 **kwargs) -> None:
        super().__init__(env)
        self.num_objectives = num_objectives
        self.features = features or self.FEATURES
        self.standard_features = self.get_standard_features(self.features)
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

    def get_standard_features(self, features):
        standard_features = features.copy()
        additional_features = ['obj_weights', 'is_controlled', 'lane_info', 'dist_ahead', 'rel_speed_ahead']
        for a in additional_features:
            if a in standard_features:
                standard_features.remove(a)
        return standard_features

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features) + 2),
            low=-np.inf, high=np.inf, dtype=np.float32
        )

    def compute_safety_features(self, ego: Vehicle) -> Tuple[float, float]:
        if not self.env.road:
            return 0.0, 0.0

        close_front = self.env.road.close_vehicles_to(ego, self.env.PERCEPTION_DISTANCE, count=1)
        if not close_front:
            return self.env.PERCEPTION_DISTANCE, 0.0

        v_front = close_front[0]
        dist = np.linalg.norm(v_front.position - ego.position)
        rel_speed = ego.speed - v_front.speed
        return dist, rel_speed

    def observe(self) -> Tuple[np.ndarray, List[Vehicle]]:
        vehicle_list = [self.observer_vehicle]

        if not self.env.road:
            return np.zeros(self.space().shape), []

        ego_dict = self.observer_vehicle.to_dict()
        valid_features = [f for f in self.standard_features if f in ego_dict.keys()]
        df = pd.DataFrame.from_records([ego_dict])[valid_features]

        dist_ahead, rel_speed_ahead = self.compute_safety_features(self.observer_vehicle)
        df["dist_ahead"] = dist_ahead
        df["rel_speed_ahead"] = rel_speed_ahead

        if "obj_weights" in self.features:
            df["objective_weights"] = np.nan
        if "is_controlled" in self.features:
            df["is_controlled"] = self.observer_vehicle.is_controlled
        if "lane_info" in self.features:
            df["lane"] = self.observer_vehicle.lane_index[2]

        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle, self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
            sort=self.order == "sorted"
        )

        if close_vehicles:
            vehicle_list.extend(close_vehicles)
            rows = []
            for v in close_vehicles[-self.vehicles_count + 1:]:
                v_dict = v.to_dict(self.observer_vehicle)
                dist_a, rel_v = self.compute_safety_features(v)
                v_dict["dist_ahead"] = dist_a
                v_dict["rel_speed_ahead"] = rel_v
                rows.append(v_dict)
            others_df = pd.DataFrame.from_records(rows)
            df = pd.concat([df, others_df], axis=0, ignore_index=True)

        if self.normalize:
            MAX_SPEED = self.observer_vehicle.MAX_SPEED
            MIN_SPEED = self.observer_vehicle.MIN_SPEED
            df = self.normalize_obs(df, MAX_SPEED, MIN_SPEED)

        if df.shape[0] < self.vehicles_count:
            pad_rows = np.zeros((self.vehicles_count - df.shape[0], len(df.columns)))
            df = pd.concat([df, pd.DataFrame(data=pad_rows, columns=df.columns)], ignore_index=True)

        obs = df.values.copy().astype(self.space().dtype)
        return obs, vehicle_list

    def normalize_obs(self, df: pd.DataFrame, MAX_SPEED, MIN_SPEED) -> pd.DataFrame:
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * MAX_SPEED, 5.0 * MAX_SPEED],
                # ✅ 改为 highway_env.road.lane.AbstractLane
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                      AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [MIN_SPEED - MAX_SPEED, MAX_SPEED - MIN_SPEED],
                "vy": [-MAX_SPEED, MAX_SPEED],
                "dist_ahead": [0, self.env.PERCEPTION_DISTANCE],
                "rel_speed_ahead": [-MAX_SPEED, MAX_SPEED]
            }

        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
            if self.clip:
                df[feature] = np.clip(df[feature], -1, 1)
        return df


class AugmentedOccupancyGridObservation(OccupancyGridObservation):
    """
    ✅ 暂时不修改 OccupancyGrid
    """
    pass
