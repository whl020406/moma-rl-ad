import gymnasium as gym
import torch
from torch import nn
from torch import device
from torch._C import device
import numpy as np

class MO_DQN:
    """ 
    Implements multi-objective DQN working with one agent. Code is based on:
    https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/single_policy/ser/mo_q_learning.py#L152
    """

    def __init__(self, env: gym.Env | None, device: device | str = "auto", seed: int | None = None, observation_space_shape: int = 10,
        num_objectives: int = 2, epsilon: float = 0.05, replay_enabled: bool = True, replay_buffer_size: int = 100, 
        batch_ratio: float = 0.1) -> None:
        
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.num_objectives = num_objectives
        self.networks = self.__create_network(np.cumsum(observation_space_shape), self.num_objectives)
        self.epsilon = epsilon
        self.replay_enabled = replay_enabled
        self.rb_size = replay_buffer_size
        self.batch_ratio = batch_ratio

        #initialise replay buffer
        self.replay_buffer = torch.zeros(size=(self.rb_size, np.cumsum(observation_space_shape)+self.num_objectives+1),
                                         device=self.device) # +1 for selected action
        self.rb_running_index = 0 #keeps track of next index of the replay buffer to be filled
        self.rb_num_elements = 0 #keeps track of the current number of elements in the replay buffer

    def __create_network(self, num_observations, num_objectives):
        #create one network for each objective
        models = []
        for _ in range(num_objectives):
            model = nn.Sequential(
                nn.Linear(num_observations+1, 128), #+1 for the action taken
                nn.ReLU(),
                nn.Linear(128, 128), #feature extractor
                nn.ReLU(),
                nn.Linear(128, 64), #feature extractor
                nn.ReLU(),
                nn.Linear(64, 1), #function approximator --> output Q value
            )
            model = model.to(self.device) #move model to desired device
            models.append(model)
        return models
    

    def __add_to_buffer(self, obs, action, reward):
        self.replay_buffer[self.rb_running_index] = torch.tensor([obs.flatten(),action, reward.flatten()], device=self.device)

        #update auxiliary variables
        self.rb_running_index = (self.rb_running_index + 1) % self.rb_size
        if self.rb_num_elements < self.rb_size:
            self.rb_num_elements += 1


    def train(self, num_iterations: int = 1000, target_update_frequency: int = 5):
        self.obs, _ = self.env.reset()

        for i in range(num_iterations):
            self.action = self.act(self.obs)
            (
                self.next_obs,
                self.reward,
                self.terminated,
                self.truncated,
                info,
            ) = self.env.step(self.action)

            self.__add_to_buffer(self.obs, self.action, self.reward)

            self.__update(i, target_update_frequency)

            #TODO: maybe use separate environment for policy evaluation

            if self.terminated or self.truncated:
                self.obs, _ = self.env.reset()
                #TODO: maybe keep track of the current number of episodes that were run

    def __update(self, current_iteration, target_update_frequency):
        #update normal network each time the function is called
        #update target network every k steps

        #fetch samples from replay buffer
        batch_indices = self.rng.choice(self.rb_num_elements, size=max(1,self.rb_num_elements*self.batch_ratio))
        batch_samples = self.replay_buffer[batch_indices]

        #update each objective network
        for obj_id in self.num_objectives:
            action_values = self.networks[obj_id].


        #update the target networks
        if (current_iteration % target_update_frequency) == 0:
            pass


    def act(self, obs):
        #choose action based on epsilon greedy policy
        pass
