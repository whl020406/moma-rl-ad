import highway_env.utils
import torch
import numpy as np
from collections import namedtuple
from typing import List, Callable
from pymoo.indicators.hv import HV
import pandas as pd
from highway_env.envs.common.observation import KinematicObservation, ObservationType
from gymnasium import spaces
from typing import List, Dict, TYPE_CHECKING, Optional, Union, Tuple
from highway_env.vehicle.kinematics import Vehicle

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


    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

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
                 num_objectives: int = 2,
                 **kwargs: dict) -> None:
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
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

    def space(self) -> spaces.Space:
        # 
        return spaces.Box(shape=(self.vehicles_count, len(self.features)+1+self.num_objectives), low=-np.inf, high=np.inf, dtype=np.float32)

    def observe(self) -> Tuple[np.ndarray, List[Vehicle]]:
        vehicle_list = []
        vehicle_list.append(self.observer_vehicle) #add current ego vehicle at first position of vehicle list
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features] #add standard features

        #add objective weights features and is_controlled flag to ego vehicle observation
        augment_dict = {"is_controlled": self.observer_vehicle.is_controlled}
        augment_dict.update({f"objective_weights_{n}": np.nan for n, w in enumerate(self.observer_vehicle.objective_weights)})
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
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features]
            
            #add additional features (objective weights and is_controlled flag for all of the close vehicles)
            augment_dict_list = [dict(
                                {"is_controlled": v.is_controlled},
                                **{f"objective_weights_{n}": float(w) for n, w in enumerate(v.objective_weights)}
                                )
                            for v in close_vehicles[-self.vehicles_count + 1:]]
            
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
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)+1+self.num_objectives))
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
    
class ChebyshevScalarisation:
    """ This class computes the chebyshev scalarisation for a vectorial Q-value and corresponding utopian point z*
        as described in Scalarized Multi-Objective Reinforcement Learning: Novel Design Techniques
        https://www.researchgate.net/publication/235698665_Scalarized_Multi-Objective_Reinforcement_Learning_Novel_Design_Techniques
        
        It acts as a non-linear alternative to linear scaling to choose actions based on vectorial Q-value estimates.
        It is implemented as a class due to the dynamic nature of the utopian point."""
    
    def __init__(self, initial_utopian: torch.Tensor, threshold_value: float, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        self.device = device
        self.z_star = initial_utopian.to(device) #initialise utopian point z*. It is a vector with the same dimensions as the vectorial Q-values
        self.threshold = threshold_value

    def scalarise_actions(self, action_q_estimates: torch.Tensor, objective_weights: torch.Tensor) -> torch.Tensor:
        action_q_estimates = torch.swapaxes(action_q_estimates,0,1) #swap axes so that rows represent q estimates of one action for all objectives
        #action_q_estimates = action_q_estimates.flatten(start_dim=0, end_dim=1)
        self.update_utopian(action_q_estimates)
        z_final = (self.z_star + self.threshold)#.reshape(-1,1)
        diffs = action_q_estimates - z_final
        abs_diffs = torch.abs(diffs)
        weighted_diffs = objective_weights * abs_diffs#.reshape(-1,1) * abs_diffs
        sq_values = torch.max(weighted_diffs, dim=1)[0]
        return sq_values

    def update_utopian(self, update_vector: torch.Tensor) -> None:
        comparison_tensor = torch.vstack([update_vector, self.z_star])
        self.z_star = torch.max(comparison_tensor, dim=0)[0]

class LinearScalarisation:

    def scalarise_actions(self, action_q_estimates, objective_weights):
        utility_values = action_q_estimates * objective_weights.reshape(-1,1)
        utility_values = torch.sum(utility_values, dim=0)
        
        return utility_values


class ReplayBuffer:
        
    def __init__(self, buffer_size, observation_space_shape, num_objectives, device, rng: np.random.Generator, importance_sampling: bool = False):
        self.size = buffer_size
        self.num_objectives = num_objectives
        self.observation_space_size = np.cumprod(observation_space_shape)[-1]
        self.device = device
        self.rng = rng
        self.importance_sampling = importance_sampling

        #initialise replay buffer
        self.buffer = torch.zeros(size=(self.size, self.observation_space_size*2+self.num_objectives+3),
                                        device=self.device) # +3 for selected action, termination flag and importance sampling id
        
        self.running_index = 0 #keeps track of next index of the replay buffer to be filled
        self.num_elements = 0 #keeps track of the current number of elements in the replay buffer

    def push(self, obs, action, next_obs, reward, terminated, importance_sampling_id = None):
        assert (not self.importance_sampling) or importance_sampling_id != None, "If importance sampling is activated, you need to provide a corresponding identifier"
        if not self.importance_sampling:
            importance_sampling_id = torch.tensor([0], device=self.device)

        elem = torch.concatenate([obs.flatten(), action, next_obs.flatten(), reward, terminated, importance_sampling_id])
        
        self.buffer[self.running_index] = elem

        #update auxiliary variables
        self.running_index = (self.running_index + 1) % self.size
        if self.num_elements < self.size:
            self.num_elements += 1

    def sample(self, sample_size):
        sample_probs = None
        if self.importance_sampling:
            sample_probs = self.compute_importance_sampling_probs()

        sample_indices = self.rng.choice(self.num_elements, p = sample_probs, size=max(1,round(sample_size)), replace=True, shuffle=True)
        return self.buffer[sample_indices]
    
    def compute_importance_sampling_probs(self):
        imp_sampling_ids = self.buffer[:self.num_elements,-1]
        min_id = torch.min(imp_sampling_ids)

        #the more recent the sample, the higher the probability of being selected
        probs = (imp_sampling_ids - min_id + 1)

        #normalise so that the sum of probs is 1
        probs = probs / torch.cumsum(probs, dim=0)[-1]
        
        probs = probs.cpu().numpy()
        return probs

    #only to be used when the samples originating from this buffer
    def get_observations(self, samples):
        return samples[:,:self.observation_space_size]

    def get_actions(self, samples: torch.Tensor):
        elem = samples[:,self.observation_space_size].to(torch.int64)#.reshape(-1,1,1) #second element was self.num_objectives
        arr = elem.repeat_interleave(repeats=self.num_objectives)
        arr = arr.reshape(-1,self.num_objectives,1)
        return arr
    def get_next_obs(self, samples):
        return samples[:,self.observation_space_size+1:self.observation_space_size*2+1]

    def get_rewards(self, samples):
        return samples[:,self.observation_space_size*2+1:-2]
    
    def get_termination_flag(self, samples):
        return samples[:,-2].flatten().to(torch.bool)
    
    def get_importance_sampling_id(self, samples):
        return samples[:,-1].flatten()


class DataLogger:
    def __init__(self, loggerName: str, fieldNames: List[str]):
        self.tupleType = namedtuple(loggerName, fieldNames)
        self.tuple_list = []

    def _add_by_list(self, entry_list: List):
        self.tuple_list.append(self.tupleType(*entry_list))
        
    def _add_by_params(self, *args, **kwargs):
        self.tuple_list.append(self.tupleType(*args, **kwargs))

    def add(self, *args, **kwargs):
        if isinstance(args, tuple) and len(args) == 1 and len(kwargs.values()) == 0:
            self._add_by_list(args[0])

        elif isinstance(args, tuple) and len(args) == 0 and len(kwargs.values()) == 1:
            self._add_by_list(list(kwargs.values())[0])

        else:
            self._add_by_params(*args, **kwargs)

    def to_dataframe(self):
        return pd.DataFrame(self.tuple_list)

def random_objective_weights(num_objectives: int, rng: np.random.Generator, device):
    random_weights = rng.random(num_objectives)
    random_weights = torch.tensor(random_weights / np.sum(random_weights), device=device) #normalise the random weights
    return random_weights


def calc_hypervolume(reference_point, reward_vector):
    '''reference point represents the worst possible value'''
    reward_vector = reward_vector * (-1) # convert to minimisation problem
    ind = HV(ref_point=reference_point)
    return ind(reward_vector)