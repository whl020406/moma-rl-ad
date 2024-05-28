import torch
import numpy as np
from collections import namedtuple
from typing import Tuple, List, NamedTuple
import pandas as pd

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
        elem = np.concatenate([obs.flatten(), [action], next_obs.flatten(), [reward], [terminated]])
        self.buffer[self.running_index] = torch.tensor(elem)
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

    def get_actions(self, samples):
        return samples[:,self.observation_space_size].to(torch.int64).reshape(-1,self.num_objectives,1)
    
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

    def add_by_list(self, entry_list: List):
        self.tuple_list.append(self.tupleType(*entry_list))

    def add(self, *args, **kwargs):
        if isinstance(args, tuple) and len(args) == 1 and len(kwargs.values()) == 0:
            self.add_by_list(args[0])

        elif isinstance(args, tuple) and len(args) == 0 and len(kwargs.values()) == 1:
            self.add_by_list(list(kwargs.values())[0])

        else:
            self.add_by_params(*args, **kwargs)
        
    def add_by_params(self, *args, **kwargs):
        self.tuple_list.append(self.tupleType(*args, **kwargs))

    def to_dataframe(self):
        return pd.DataFrame(self.tuple_list)

def random_objective_weights(num_objectives: int, rng: np.random.Generator):
    random_weights = rng.random(num_objectives)
    random_weights = random_weights / np.sum(random_weights) #normalise the random weights
    return random_weights