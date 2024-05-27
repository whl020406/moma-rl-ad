import torch
import numpy as np

class ReplayBuffer:
        
    def __init__(self, buffer_size, observation_space_shape, num_objectives, device, rng: np.random.Generator):
        self.size = buffer_size
        self.num_objectives = num_objectives
        self.observation_space_size = np.cumsum(observation_space_shape)
        self.device = device
        self.rng = rng

        #initialise replay buffer
        self.buffer = torch.zeros(size=(self.size, self.observation_space_size*2+self.num_objectives+2),
                                        device=self.device) # +2 for selected action and termination flag
        
        self.running_index = 0 #keeps track of next index of the replay buffer to be filled
        self.num_elements = 0 #keeps track of the current number of elements in the replay buffer
    
    def push(self, obs, action, next_obs, reward, terminated):
        self.buffer[self.running_index] = torch.tensor([obs.flatten(), action, next_obs.flatten(), reward.flatten(), terminated], device=self.device)
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
        return samples[:,self.observation_space_size]
    
    def get_next_obs(self, samples):
        return samples[:,self.observation_space_size+1:self.observation_space_size*2+1]

    def get_rewards(self, samples):
        return samples[:,self.observation_space_size*2+1:-1]
    
    def get_termination_flag(self, samples):
        return samples[:,-1].flatten()