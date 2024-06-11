import gymnasium as gym
import torch
from torch import nn
from torch import device
import numpy as np
from typing import Tuple, Sequence
from src.utils import ReplayBuffer, random_objective_weights, DataLogger
from torch.nn.modules.loss import _Loss
from tqdm import trange
from typing import List
class DQN_Network(nn.Module):

    def __init__(self, n_observations, n_actions, n_objectives):
        super(DQN_Network, self).__init__()
        self.layer1 = nn.ModuleList([nn.Linear(n_observations, 128) for _ in range(n_objectives)])
        self.layer2 = nn.ModuleList([nn.Linear(128, 128) for _ in range(n_objectives)])
        self.layer3 = nn.ModuleList([nn.Linear(128, n_actions) for _ in range(n_objectives)])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        n_objectives = len(self.layer1)
        #x = torch.tensor(x.flatten())
        x_arr = [torch.relu(self.layer1[i](x)) for i in range(n_objectives)]
        x_arr = [torch.relu(self.layer2[i](x_arr[i])) for i in range(n_objectives)]
        x_arr = [self.layer3[i](x_arr[i]) for i in range(n_objectives)]
        stacked_arr = torch.stack(x_arr, dim=1)
        return stacked_arr



class MO_DQN:
    """ 
    Implements multi-objective DQN working with one agent. Code is based on:
    https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/single_policy/ser/mo_q_learning.py#L152
    """

    def __init__(self, env: gym.Env | None, device: device = None, seed: int | None = None, 
        observation_space_shape: Sequence[int] = [1,1], num_objectives: int = 2, num_actions: int = 5, epsilon: float = 0.05, 
        replay_enabled: bool = True, replay_buffer_size: int = 100, batch_ratio: float = 0.1, objective_weights: Sequence[float] = None,
        optimiser: torch.optim.Optimizer = torch.optim.SGD, loss_criterion: _Loss = nn.SmoothL1Loss, episode_recording_interval: int = None,
        objective_names: List[str] = None) -> None:
        
        if objective_names is None:
            objective_names = [f"reward_{x}" for x in range(num_objectives)]
        
        assert len(objective_names) == num_objectives, "The number of elements in the objective_names list must be equal to the number of objectives!"

        self.env = env
        self.recording_interval = episode_recording_interval
        if episode_recording_interval is not None:
            self.env = gym.wrappers.RecordVideo(self.env, video_folder="videos", name_prefix="training_MODQN", 
                                                episode_trigger=lambda x: x % self.recording_interval == 0)
            self.env.metadata["render_fps"] = 30
            
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_objectives = num_objectives
        self.objective_weights = objective_weights
        if self.objective_weights is None:
            self.objective_weights = random_objective_weights(self.num_objectives, self.rng, self.device)

        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape

        (self.policy_net, self.target_net) = \
        self.__create_network(np.cumprod(observation_space_shape)[-1], self.num_actions, self.num_objectives)

        self.epsilon = epsilon
        self.replay_enabled = replay_enabled
        self.rb_size = replay_buffer_size
        self.batch_ratio = batch_ratio

        self.optimiser_class = optimiser
        self.loss_criterion = loss_criterion

        #initialise replay buffer
        self.buffer = ReplayBuffer(self.rb_size, observation_space_shape, self.num_objectives, self.device, self.rng)

        #initialise reward logger
        feature_names = ["episode"]
        feature_names.extend(objective_names)
        self.reward_logger = DataLogger("reward_logger",feature_names)


    def __create_network(self, num_observations, num_actions, num_objectives) -> Tuple[nn.Module, nn.Module]:
        #create one network for each objective
        policy_net = DQN_Network(num_observations, num_actions, num_objectives).to(self.device)
        target_net = DQN_Network(num_observations, num_actions, num_objectives).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())

        return policy_net, target_net

    def train(self, num_iterations: int = 1000, target_update_frequency: int = 5, gamma: float = 1):
        self.obs, _ = self.env.reset()
        self.obs = torch.tensor(self.obs[0].reshape(1,-1), device=self.device) #TODO: remove when going to multi-agent
        self.gamma = gamma
        self.optimiser = self.optimiser_class(self.policy_net.parameters())
        self.loss_func = self.loss_criterion()
        accumulated_rewards = np.zeros(self.num_objectives)
        episode_nr = 0
        #take step in environment
        for i in trange(num_iterations, desc="Iterations", mininterval=2):
            self.action = self.act(self.obs, eps_greedy=True)
            (
                self.next_obs,
                self.reward,
                self.terminated,
                self.truncated,
                info,
            ) = self.env.step(self.action)
            self.next_obs = self.next_obs[0] #TODO: remove when going to multi-agent
            #accumulate reward
            accumulated_rewards += self.reward
            #push to replay buffer
            self.buffer.push(self.obs, self.action, self.next_obs, self.reward, self.terminated)

            #update the weights
            self.__update_weights(i, target_update_frequency)

            if self.terminated or self.truncated:
                self.reward_logger.add(episode_nr, *list(accumulated_rewards))

                accumulated_rewards = 0
                episode_nr += 1
                self.obs, _ = self.env.reset()
                self.obs = torch.tensor(self.obs[0].reshape(1,-1), device=self.device) #TODO: remove when going to multi-agent
                self.objective_weights = random_objective_weights(self.num_objectives, self.rng, self.device)

        return self.reward_logger.to_dataframe()

    def __update_weights(self, current_iteration, target_update_frequency):
        #update normal network each time the function is called
        #update target network every k steps
        self.policy_net.train()
        self.target_net.eval()
        #fetch samples from replay buffer
        batch_samples = self.buffer.sample(round(self.buffer.num_elements*self.batch_ratio))
        observations = self.buffer.get_observations(batch_samples)
        next_obs = self.buffer.get_next_obs(batch_samples)
        actions = self.buffer.get_actions(batch_samples)
        term_flags = self.buffer.get_termination_flag(batch_samples)
        rewards  = self.buffer.get_rewards(batch_samples)
        #go through each sample of the batch
        #fetch Q values of the current observation and action from all the objectives Q-networks
        #if (observations.shape[0] > 1):
        #    print(observations.shape)
        state_action_values = self.policy_net(observations)
        state_action_values = state_action_values.gather(2, actions)
        state_action_values = state_action_values.reshape(observations.shape[0],self.num_objectives)
        next_state_values = self.target_net(next_obs).max(2).values
        next_state_values[term_flags] = 0
            
        exp_state_action_values = next_state_values * self.gamma + rewards
        #compute loss between estimates and actual values
        loss = self.loss_func(state_action_values, exp_state_action_values)

        #backpropagate loss
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        #update the target networks
        if (current_iteration % target_update_frequency) == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
    def act(self, obs, eps_greedy: bool = False):
        #TODO: during execution: only select based on available actions instead of all actions when eps_greedy is false
        #choose action based on epsilon greedy policy and policy network
        #assumption: actions are discrete and labelled from 0 to n-1
        r = self.rng.random()
        action = None

        #select best action according to policy
        if not eps_greedy or r > self.epsilon:
            with torch.no_grad():
                self.policy_net.eval()
                q_values = self.policy_net(obs)
                utility_values = q_values * self.objective_weights.reshape(-1,1)
                utility_values = torch.sum(utility_values, dim=1)
                action = torch.argmax(utility_values).item()

        else: # choose random action
            action = self.rng.choice(self.num_actions)

        return action
    
    def set_objective_weights(self, weights: np.ndarray):
        self.objective_weights = weights
