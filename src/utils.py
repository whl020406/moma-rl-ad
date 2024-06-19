import torch
import numpy as np
from collections import namedtuple
from typing import List
from highway_env.vehicle.kinematics import Vehicle
import pandas as pd


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
        self.update_utopian(action_q_estimates.reshape(-1,self.z_star.shape[0])) #TODO: test if that line results in the action_q_estimates to take on the correct shape
        z_final = (self.z_star + self.threshold).reshape(-1,1)
        diffs = action_q_estimates - z_final
        abs_diffs = torch.abs(diffs)
        weighted_diffs = objective_weights.reshape(-1,1) * abs_diffs
        sq_values = torch.max(weighted_diffs, dim=1)[0]
        return sq_values

    def update_utopian(self, update_vector: torch.Tensor) -> None:
        comparison_tensor = torch.vstack([update_vector, self.z_star])
        self.z_star = torch.max(comparison_tensor, dim=0)[0]

class LinearScalarisation:

    def scalarise_actions(self, action_q_estimates, objective_weights):
        utility_values = action_q_estimates * objective_weights.reshape(-1,1)
        utility_values = torch.sum(utility_values, dim=1)
        
        return utility_values


class ReplayBuffer:
        
    def __init__(self, buffer_size, observation_space_shape, num_objectives, device, rng: np.random.Generator):
        self.size = buffer_size
        self.num_objectives = num_objectives
        self.observation_space_size = np.cumprod(observation_space_shape)[-1]
        self.device = device
        self.rng = rng

        #initialise replay buffer
        self.buffer = torch.zeros(size=(self.size, self.observation_space_size*2+self.num_objectives+2),
                                        device=self.device) # +2 for selected action and termination flag
        
        self.running_index = 0 #keeps track of next index of the replay buffer to be filled
        self.num_elements = 0 #keeps track of the current number of elements in the replay buffer
    
    def push(self, obs, action, next_obs, reward, terminated):
        elem = torch.concatenate([obs.flatten(), action, next_obs.flatten(), reward, terminated])
        self.buffer[self.running_index] = elem
        #update auxiliary variables
        self.running_index = (self.running_index + 1) % self.size
        if self.num_elements < self.size:
            self.num_elements += 1

    def sample(self, sample_size):
        sample_indices = self.rng.choice(self.num_elements, size=max(1,sample_size), replace=True, shuffle=True)
        return self.buffer[sample_indices]
    

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
        return samples[:,self.observation_space_size*2+1:-1]
    
    def get_termination_flag(self, samples):
        return samples[:,-1].flatten().to(torch.bool)


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


def calc_energy_efficiency(vehicle: Vehicle, type='light_passenger',fuel='gasoline', normalise: bool = False):
    ''' Calculates the difference between maximum and current CO2 emmissions, 
        which is taken as a means of measuring the energy efficiency.
        Meaning: poor energy efficiency --> close to 0, great efficiency --> close to value 1 (for normalise = True)
    '''

    curr_ac = vehicle.action['acceleration'] 
    curr_speed = vehicle.speed

    # maximum possible acceleration computed using the formula found in ControlledVehicle class of highway-env
    max_acc = vehicle.KP_A * (vehicle.MAX_SPEED - vehicle.MIN_SPEED)
    max_speed = vehicle.MAX_SPEED

    max_energy_consumption = compute_co2_emission(max_acc, max_speed)
    current_energy_consumption = compute_co2_emission(curr_ac, curr_speed)

    efficiency = max_energy_consumption - current_energy_consumption
    
    # compute maximum possible energy consumption based on maximum velocity and acceleration 
    # and use this value for normalisation
    if normalise:
        efficiency /= max_energy_consumption

    return efficiency

def compute_co2_emission(acceleration, velocity, type='light_passenger',fuel='gasoline'):
    '''Code taken from: https://github.com/amrzr/SA-MOEAMOPG/blob/55ceddc58062f2d7d26107d7813d2dd7328f2203/SAMOEA_PGMORL/highway_env/envs/two_way_env.py#L139C1-L209C22.
        Equation is described in: https://journals.sagepub.com/doi/full/10.1177/0361198119839970'''
     
    if fuel == 'gasoline':
        T_idle = 2392    # CO2 emission from gasoline [gCO2/L]
        E_gas =  31.5e6  # Energy in gasoline [J\L]
    elif fuel == 'diesel':
        T_idle = 2660   # CO2 emission from diesel [gCO2/L]
        E_gas =  38e6   # Energy in diesel [J\L]

    if type == 'light_passenger':
        M = 1334    # light-duty passenger vehicle mass [kg]
    elif type == 'light_van':
        M = 1752    # light-duty van vehicle mass [kg]
    
    Crr = 0.015     # Rolling resistance
    Cd  = 0.3       # Aerodynamic drag coefficient
    A = 2.5         # Frontal area [m2]
    g = 9.81        # Gravitational acceleration
    r = 0           # Regeneration efficiency ratio
    pho = 1.225     # Air density
    fuel_eff = 0.7  # fuel efficiency [70%]

    
    condition = M  * acceleration * velocity + M  * g * Crr * velocity +0.5 * Cd * A  * pho * velocity **3
    
    Ei = T_idle  / E_gas  * condition

    if Ei <= 0:
        E = r
    else:
        Ei = Ei * (velocity + 0.5 * acceleration)
        E = Ei/fuel_eff

    return np.abs(E)