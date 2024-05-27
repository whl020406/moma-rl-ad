import gymnasium as gym
import torch
from torch import nn
from torch import device
import numpy as np
from ReplayBuffer import ReplayBuffer

class DQN_Network(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class MO_DQN:
    """ 
    Implements multi-objective DQN working with one agent. Code is based on:
    https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/single_policy/ser/mo_q_learning.py#L152
    """

    def __init__(self, env: gym.Env | None, device: device | str = "auto", seed: int | None = None, 
        observation_space_shape: int = 10, num_objectives: int = 2, num_actions: int = 5, epsilon: float = 0.05, 
        replay_enabled: bool = True, replay_buffer_size: int = 100, batch_ratio: float = 0.1) -> None:
        
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.num_objectives = num_objectives
        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape

        (self.policy_nets, self.target_nets) = \
        self.__create_network(np.cumsum(observation_space_shape), self.num_objectives, self.num_actions)

        self.epsilon = epsilon
        self.replay_enabled = replay_enabled
        self.rb_size = replay_buffer_size
        self.batch_ratio = batch_ratio

        #initialise replay buffer
        self.replay_buffer = ReplayBuffer(self.rb_size, observation_space_shape, self.num_objectives, self.device)
        

    def __create_network(self, num_observations, num_objectives, num_actions):
        #create one network for each objective
        policy_nets = []
        target_nets = []
        for _ in range(num_objectives):
            p_net = DQN_Network(num_observations, num_actions).to(self.device)
            t_net = DQN_Network(num_observations, num_actions).to(self.device)
            t_net.load_state_dict(p_net.state_dict())

            policy_nets.append(p_net)
            target_nets.append(t_net)

        return policy_nets, target_nets

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

        #go through each sample of the batch
        for sample_id in range(batch_indices.shape[0]):
            #fetch Q values of the current observation and action from all the objectives Q-networks
            #compute loss between estimates and actual values
            #backpropagate loss
            pass


        #update the target networks
        if (current_iteration % target_update_frequency) == 0:
            pass


    def act(self, obs):
        #choose action based on epsilon greedy policy
        pass
