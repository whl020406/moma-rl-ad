import highway_env.utils
import torch
import numpy as np
from typing import List
import pandas as pd
from highway_env.envs.common.observation import KinematicObservation, OccupancyGridObservation, ObservationType
from gymnasium import spaces
from typing import List, Dict, TYPE_CHECKING, Tuple
from highway_env.vehicle.kinematics import Vehicle, Optional
from highway_env import utils

import highway_env
from highway_env.road.lane import AbstractLane

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class AugmentedMultiAgentObservation(ObservationType):
    '''Extends the MultiAgentObservation class to work with AugmentedKinematicObservation.
       Code is taken directly from the MultiAgentObservation class featured in highway_env and is 
       slightly modified by explicitly setting the obs_type to AugmentedKinematicObservation.'''
    
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
                    case _: #default
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
        
        #to not alter the api, this variable has to be accessed within the reward function to fetch the properties of the other vehicles
        self.curr_observation_vehicle_lists = vehicle_lists
        return tuple(observations)


class AugmentedKinematicObservation(KinematicObservation):
    '''Extends the KinematicObservation class to include the objective weights in a two-objective setting.
       Code is taken directly from the KinematicObservation class featured in highway_env and is modified at 
       various points.'''


    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy', "lane_info"]

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
                 num_objectives: int = 2, **kwargs) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        :param num_objectives: number of objectives whose weights to include in the observation
        """
        super().__init__(env)
        self.num_objectives = num_objectives #add num objectives for use in space function
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
        additional_features = ['obj_weights', 'lane_info']
        for a in additional_features:
            if a in standard_features:
                standard_features.remove(a)

        return standard_features
    def space(self) -> spaces.Space:
        self.num_features = len(self.features)
        if "obj_weights" in self.features:
            self.num_features += self.num_objectives
        return spaces.Box(shape=(self.vehicles_count, self.num_features), low=-np.inf, high=np.inf, dtype=np.float32)

    def observe(self) -> Tuple[np.ndarray, List[Vehicle]]:
        vehicle_list = []
        augment_dict = {}
        vehicle_list.append(self.observer_vehicle) #add current ego vehicle at first position of vehicle list
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.standard_features] #add standard features

        #add objective weights features and is_controlled flag to ego vehicle observation
        if "obj_weights" in self.features:
            augment_dict.update({"is_controlled": self.observer_vehicle.is_controlled})
            augment_dict.update({f"objective_weights_{n}": np.nan for n, w in enumerate(self.observer_vehicle.objective_weights)})
        if "lane_info" in self.features:
            augment_dict.update({"lane": self.observer_vehicle.lane_index[2]})

        augment_df = pd.DataFrame.from_records([augment_dict])
        df = pd.concat([df,augment_df], axis=1)
        
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            vehicle_list.extend(close_vehicles) #add close vehicles to vehicle list
            origin = self.observer_vehicle if not self.absolute else None
            others_standard_df = pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.standard_features]
            
            #add additional features (objective weights and is_controlled flag for all of the close vehicles)
            augment_dict_list = []
            for v in close_vehicles[-self.vehicles_count + 1:]:
                v_dict = {}
                if "obj_weights" in self.features:
                    v_dict.update(dict(
                                {"is_controlled": v.is_controlled},
                                **{f"objective_weights_{n}": float(w) for n, w in enumerate(v.objective_weights)}
                                ))
                if "lane_info" in self.features:
                    v_dict["lane"] = v.lane_index[2]
                
                augment_dict_list.append(v_dict)
            
            others_augment_df = pd.DataFrame.from_records(augment_dict_list)

            #append observations of nearby vehicles to dataframe of ego vehicle observations
            others_df = pd.concat([others_standard_df, others_augment_df], axis = 1)
            df = pd.concat([df,others_df], axis=0, ignore_index=True)
        
        # Normalize and clip
        if self.normalize:
            MAX_SPEED = self.observer_vehicle.MAX_SPEED
            MIN_SPEED = self.observer_vehicle.MIN_SPEED
            df = self.normalize_obs(df, MAX_SPEED, MIN_SPEED)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], self.num_features))
            df = pd.concat([df, pd.DataFrame(data=rows, columns=df.columns)], ignore_index=True)
        # Reorder
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        obs = obs.astype(self.space().dtype)
        return obs.astype(self.space().dtype), vehicle_list
    
    def normalize_obs(self, df: pd.DataFrame,MAX_SPEED, MIN_SPEED) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * MAX_SPEED, 5.0 * MAX_SPEED],
                "y": [- AbstractLane.DEFAULT_WIDTH* len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [MIN_SPEED - MAX_SPEED, MAX_SPEED - MIN_SPEED],
                "vy": [-MAX_SPEED, MAX_SPEED]
            }
        vx_range_ego = [MIN_SPEED, MAX_SPEED]
        for feature, f_range in self.features_range.items():
            if feature in df:
                    #if feature is velocity in x direction
                    if feature == "vx":
                        #do different normalisation for ego vehicle
                        df.loc[0,feature] = highway_env.utils.lmap(df.loc[0,feature],vx_range_ego, [0, 1])
                        #and the standard normalisation for other vehicles
                        df.loc[1:,feature] = highway_env.utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                    else:
                        df[feature] = highway_env.utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
            if self.clip:
                df[feature] = np.clip(df[feature], -1, 1)
        return df


class AugmentedOccupancyGridObservation(OccupancyGridObservation):

    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'vx', 'vy', 'on_road' ,"obj_weights", "is_controlled"]
    GRID_SIZE: List[List[float]] = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP: List[int] = [5, 5]

    def __init__(self,
                 env: 'AbstractEnv',
                 features: Optional[List[str]] = None,
                 grid_size: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 grid_step: Optional[Tuple[float, float]] = None,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 align_to_vehicle_axes: bool = False,
                 clip: bool = True,
                 as_image: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param grid_size: real world size of the grid [[min_x, max_x], [min_y, max_y]]
        :param grid_step: steps between two cells of the grid [step_x, step_y]
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: use absolute or relative coordinates
        :param align_to_vehicle_axes: if True, the grid axes are aligned with vehicle axes. Else, they are aligned
               with world axes.
        :param clip: clip the observation in [-1, 1]
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        self.grid_step = np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step),
                                dtype=np.uint8)
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image

    def space(self) -> spaces.Space:
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape), []

        if self.absolute:
            raise NotImplementedError()
        else:
            # construct list of close vehicles
            vehicle_list = []
            vehicle_list.append(self.observer_vehicle)

            # Add nearby traffic
            close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
            if close_vehicles:
                vehicle_list.extend(close_vehicles) #add close vehicles to vehicle list

            # populate the grid
            self.grid.fill(np.nan)

            # Get nearby traffic data
            vehicle_dicts = []
            for vehicle in self.env.road.vehicles:
                v_dict = vehicle.to_dict(self.observer_vehicle) #dict with standard features

                #adding additional features for the augmented observation space
                if "obj_weights" in self.features:
                    v_dict["obj_weights"] = vehicle.obj_weights
                if "is_controlled" in self.features:
                    v_dict["is_controlled"] = vehicle.is_controlled
                
                vehicle_dicts.append(v_dict)

            df = pd.DataFrame.from_records(vehicle_dicts)

            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # A vehicle feature
                    for _, vehicle in df[::-1].iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # Recover unnormalized coordinates for cell index
                        if "x" in self.features_range:
                            x = utils.lmap(x, [-1, 1], [self.features_range["x"][0], self.features_range["x"][1]])
                        if "y" in self.features_range:
                            y = utils.lmap(y, [-1, 1], [self.features_range["y"][0], self.features_range["y"][1]])
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            self.grid[layer, cell[1], cell[0]] = vehicle[feature]

                elif feature == "on_road":
                    self.fill_road_layer_by_lanes(layer)

            obs = self.grid

            if self.clip:
                obs = np.clip(obs, -1, 1)

            if self.as_image:
                obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            obs = np.nan_to_num(obs).astype(self.space().dtype)

            return obs, vehicle_list