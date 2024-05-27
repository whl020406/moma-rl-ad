import gymnasium as gym
import torch
from torch import nn
from torch import device
import numpy as np
from typing import Tuple
from ReplayBuffer import ReplayBuffer

class DQN_Network(nn.Module):

    def __init__(self, n_observations, n_actions, n_objectives):
        super(DQN_Network, self).__init__()
        self.layer1 = [nn.Linear(n_observations, 128) for _ in range(n_objectives)]
        self.layer2 = [nn.Linear(128, 128) for _ in range(n_objectives)]
        self.layer3 = [nn.Linear(128, n_actions) for _ in range(n_objectives)]

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        n_objectives = len(self.layer1)
        x_arr = [torch.relu(self.layer1(x)) for _ in range(n_objectives)]
        x_arr = [torch.relu(self.layer2(x_arr[i])) for i in range(n_objectives)]
        return torch.tensor([self.layer3(x_arr[i]) for i in range(n_objectives)])


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

        (self.policy_net, self.target_net) = \
        self.__create_network(np.cumsum(observation_space_shape), self.num_actions, self.num_objectives)

        self.epsilon = epsilon
        self.replay_enabled = replay_enabled
        self.rb_size = replay_buffer_size
        self.batch_ratio = batch_ratio

        #initialise replay buffer
        self.buffer = ReplayBuffer(self.rb_size, observation_space_shape, self.num_objectives, self.device, self.rng)

    def __create_network(self, num_observations, num_objectives, num_actions) -> Tuple[nn.Module, nn.Module]:
        #create one network for each objective
        policy_net = DQN_Network(num_observations, num_actions, num_objectives).to(self.device)
        target_net = DQN_Network(num_observations, num_actions, num_objectives).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())

        return policy_net, target_net

    def train(self, num_iterations: int = 1000, target_update_frequency: int = 5, gamma: float = 1):
        self.obs, _ = self.env.reset()
        self.gamma = gamma

        #take step in environment
        for i in range(num_iterations):
            self.action = self.act(self.obs)
            (
                self.next_obs,
                self.reward,
                self.terminated,
                self.truncated,
                info,
            ) = self.env.step(self.action)

            #push to replay buffer
            self.buffer.push(self.obs, self.action, self.reward, self.terminated)

            #update the weights
            self.__update_weights(i, target_update_frequency)

            #TODO: maybe use separate environment for policy evaluation

            if self.terminated or self.truncated:
                self.obs, _ = self.env.reset()
                #TODO: maybe keep track of the current number of episodes that were run

    def __update_weights(self, current_iteration, target_update_frequency):
        #update normal network each time the function is called
        #update target network every k steps

        #fetch samples from replay buffer
        batch_samples = self.buffer.sample(self.rb_num_elements*self.batch_ratio)
        observations = self.buffer.get_observations(batch_samples)
        next_obs = self.buffer.get_next_obs(batch_samples)
        actions = self.buffer.get_actions(batch_samples)
        term_flags = self.buffer.get_termination_flag(batch_samples)
        rewards  = self.buffer.get_rewards(batch_samples)
        #go through each sample of the batch
        with torch.no_grad():
            #fetch Q values of the current observation and action from all the objectives Q-networks
            state_action_values = torch.tensor(self.policy_net(observations).gather(1, actions), device=self.device)
            next_state_values = torch.tensor(self.target_net(next_obs).max(1).values, device=self.device)
            next_state_values[term_flags] = 0
            
        exp_state_action_values = next_state_values * self.gamma + rewards
        #compute loss between estimates and actual values
        loss = nn.SmoothL1Loss(state_action_values, exp_state_action_values)

        #backpropagate loss
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        #update the target networks
        if (current_iteration % target_update_frequency) == 0:
            for i in range(len(self.target_nets)):
                self.target_nets[i].set_state_dict(self.policy_nets[i].state_dict())

    def act(self, obs, eps_greedy: bool = False):
        #choose action based on epsilon greedy policy and policy network
        r = self.rng.random()
        action = None
        
        #select best action according to policy
        if eps_greedy and r > self.epsilon:
            with torch.no_grad():
                action = self.policy_net(obs).max(1).values

        else: # choose random action
            action = self.rng.choice(self.num_actions)

        return action

