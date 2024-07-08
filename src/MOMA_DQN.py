import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0
import torch
from torch import nn
from torch import device
import numpy as np
from typing import Tuple, Sequence
from src.utils import ReplayBuffer, random_objective_weights, DataLogger, LinearScalarisation
from torch.nn.modules.loss import _Loss
from tqdm import trange
from typing import List
from pymoo.util.ref_dirs import get_reference_directions
from DQN_Network import DQN_Network
from utils import AugmentedMultiAgentObservation

class MOMA_DQN:
    """ 
    Implements multi-objective multi-agent DQN by extending the code in MO_DQN.py
    It works by sharing the parameters between agents, adjusting the reward function to include a social and egoistisc term
    and sharing experiences using a shared replay buffer. The DQN network has access to the objective weights of other autonomous agents
    and uses a fixed weight of human agents.
    """

    def __init__(self, env: gym.Env | None, device: device = None, seed: int | None = None, 
        observation_space_length: int = 30, num_objectives: int = 2, num_actions: int = 5, 
        replay_enabled: bool = True, replay_buffer_size: int = 1000, batch_ratio: float = 0.2, objective_weights: Sequence[float] = None,
        loss_criterion: _Loss = nn.SmoothL1Loss, observation_space_type = AugmentedMultiAgentObservation,
        objective_names: List[str] = None, scalarisation_method = LinearScalarisation, scalarisation_argument_list: List = [],
        ego_reward_priority: float = 0.5, separate_ego_and_social_reward: bool = True) -> None:
        
        if objective_names is None:
            objective_names = [f"reward_{x}" for x in range(num_objectives)]
        
        assert len(objective_names) == num_objectives, "The number of elements in the objective_names list must be equal to the number of objectives!"
        self.objective_names = objective_names
        self.ego_reward_priority = ego_reward_priority
        self.separate_ego_and_social_reward = separate_ego_and_social_reward
        self.env = env
        self.num_controlled_vehicles = len(self.env.unwrapped.controlled_vehicles)
        self.observation_space_type = observation_space_type
            
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

        self.observation_space_length = observation_space_length

        #minus term because the objective weights of the ego vehicle are excluded
        (self.policy_net, self.target_net) = \
        self.__create_network(self.observation_space_length, self.num_actions, self.num_objectives)

        self.replay_enabled = replay_enabled
        self.rb_size = replay_buffer_size
        self.batch_ratio = batch_ratio

        self.loss_criterion = loss_criterion

        #initialise replay buffer
        #*2 for num_objectives because we want to store the rewards of the ego vehicle and the mean reward of close vehicles separately
        self.buffer = ReplayBuffer(self.rb_size, self.observation_space_length, self.num_objectives*2, self.device, self.rng, importance_sampling=True)

        #initialise reward logger
        feature_names = ["episode"]
        feature_names.extend(self.objective_names)
        self.reward_logger = DataLogger("reward_logger",feature_names)

        #initialise scalarisation function
        self.scalarisation_method = scalarisation_method(*scalarisation_argument_list)


    def __create_network(self, num_observations, num_actions, num_objectives) -> Tuple[nn.Module, nn.Module]:
            #create one network for each objective
            policy_net = DQN_Network(num_observations, num_actions, num_objectives).to(self.device)
            target_net = DQN_Network(num_observations, num_actions, num_objectives).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())

            return policy_net, target_net

    def train(self, num_episodes: int = 5_000, inv_optimisation_frequency: int = 1, inv_target_update_frequency: int = 5, 
                gamma: float = 0.9, epsilon_start: float = 0.1, epsilon_end: float = 0) :
        '''
        Runs the training procedure for num_iterations iterations. The inv_optimisation_frequency specifies 
        the number of iterations after which a weight update occurs.The inv_target_update_frequency specifies 
        the number of weight updates of the policy net, after which the target net weights are adjusted.
        Gamma is the discount factor for the rewards. Epsilon is the probability of a random action being selected during training.
        Its value is linearly reduced during the training procedure from epsilon_start to epsilon_end.
        '''
        self.obs, _ = self.env.reset()
        self.obs = [torch.tensor(single_obs, device=self.device) for single_obs in self.obs] #reshape observations and
        self.obs = [single_obs[~torch.isnan(single_obs)].reshape(1,-1) for single_obs in self.obs] #remove nan values
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.optimiser = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)

        self.loss_func = self.loss_criterion()
        accumulated_rewards = np.zeros(self.num_objectives)
        episode_nr = 0
        num_of_conducted_optimisation_steps = 0
        #take step in environment
        for episode_nr in trange(num_episodes, desc="Training episodes", mininterval=2):
            self.terminated = False
            self.truncated = False
            while not (self.terminated or self.truncated):
                self.actions = self.act(self.obs, eps_greedy=True)
                (
                    self.next_obs,
                    self.rewards,
                    self.terminated,
                    self.truncated,
                    info,
                ) = self.env.step(self.actions)
                
                self.crashed = info["crashed"]
                self.vehicle_obj_weights = info["vehicle_objective_weights"]
                #accumulate episode reward

                #TODO: figure out what data to log: look at corresponding trello entry

                self.next_obs = [torch.tensor(single_obs, device=self.device) for single_obs in self.next_obs] #reshape observations and
                self.next_obs = [single_obs[~torch.isnan(single_obs)].reshape(1,-1) for single_obs in self.next_obs] #remove nan values
                
                reward_summary = self.compute_reward_summary(self.rewards, self.vehicle_obj_weights)

                self.buffer.push(self.obs, self.actions, self.next_obs, reward_summary, self.crashed, episode_nr, num_samples=self.num_controlled_vehicles)
                
                #use next_obs as obs during the next iteration
                self.obs = self.next_obs

                #update the weights every optimisation_frequency steps and only once the replay buffer is filled
                if ((episode_nr % inv_optimisation_frequency) == 0) and (self.buffer.num_elements == self.rb_size):

                    self.__update_weights(num_of_conducted_optimisation_steps, inv_target_update_frequency)
                    num_of_conducted_optimisation_steps += 1

            #reset environment if it was terminated
            if self.terminated or self.truncated:
                self.reduce_epsilon(num_episodes, epsilon_start, epsilon_end) #linearly reduce the value of epsilon
                self.reward_logger.add(episode_nr, *list(accumulated_rewards))

                accumulated_rewards = np.zeros(self.num_objectives)
                self.obs, _ = self.env.reset()
                self.obs = [torch.tensor(single_obs, device=self.device) for single_obs in self.obs] #reshape observations and
                self.obs = [single_obs[~torch.isnan(single_obs)].reshape(1,-1) for single_obs in self.obs] #remove nan values
                self.objective_weights = random_objective_weights(self.num_objectives, self.rng, self.device)

        return self.reward_logger.to_dataframe()

    def compute_reward_summary(self, rewards, obj_weights):
        #TODO: debug shape mismatch
        reward_summary = []

        for i in range(self.num_controlled_vehicles):
            r = torch.from_numpy(rewards[i]).to(self.device)
            weights = torch.stack(obj_weights[i], dim=0).to(self.device)
            weights[torch.all(weights == 0, dim=1)] = 1 #where vehicles are not controlled --> add raw rewards
            
            #this can happen when one of the next vehicle is far away from ego
            #thus we need to bring it to the same shape
            if r.shape[0] > weights.shape[0]:
                assert torch.all(r[weights.shape[0]+1:] == -1) #make sure all rewards beyond this point are -1
                r = r[:weights.shape[0]] #remove all excess rows from r
            
            weighted_reward = r * weights
            ego_reward = weighted_reward[0,:]
            if weighted_reward.shape[0] > 1:
                social_reward = weighted_reward[1:,:]
                mean_social_reward = torch.sum(social_reward,dim=0) / (social_reward.shape[0]) #-1 to exclude the ego vehicle
            else:
                mean_social_reward = torch.tensor([0,0], device=self.device) #when no other vehicles are around the ego vehicle
            #this means, in replay buffer, the first num_objectives elements in the reward are the ego reward 
            #and the other two are the mean social reward
            reward_summary.append(torch.hstack([ego_reward, mean_social_reward]))
        
        return reward_summary
            

    def __update_weights(self, current_optimisation_iteration, inv_target_update_frequency):
        #TODO: adjust to work in MOMA setting

        #fetch samples from replay buffer
        batch_samples = self.buffer.sample(round(self.buffer.num_elements*self.batch_ratio))
        observations = self.buffer.get_observations(batch_samples)
        next_obs = self.buffer.get_next_obs(batch_samples)
        actions = self.buffer.get_actions(batch_samples)
        term_flags = self.buffer.get_termination_flag(batch_samples)
        rewards  = self.buffer.get_rewards(batch_samples)
        #go through each sample of the batch
        #fetch Q values of the current observation and action from all the objectives Q-networks
        
        if self.separate_ego_and_social_reward:
            actions = actions[:,0:self.num_objectives,:]

        state_action_values = self.policy_net(observations)
        state_action_values = state_action_values.gather(2, actions)
        state_action_values = state_action_values.reshape(observations.shape[0],self.num_objectives)
        
        with torch.no_grad():
            next_state_values = self.target_net(next_obs).max(2).values
        next_state_values[term_flags] = 0

        ego_rewards = rewards[:,0:self.num_objectives]
        mean_social_rewards = rewards[:,self.num_objectives:]

        #uses ego and mean social rewards
        exp_state_action_values = next_state_values * self.gamma + \
        (ego_rewards * self.ego_reward_priority + mean_social_rewards * (1-self.ego_reward_priority))

        #compute loss between estimates and actual values
        loss = self.loss_func(state_action_values, exp_state_action_values)
        #backpropagate loss
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimiser.step()

        #update the target networks
        if (current_optimisation_iteration % inv_target_update_frequency) == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


    def act(self, obs, eps_greedy: bool = False):
        '''select a list of actions, one element for each autonomously controlled agent'''
        joint_action = []
        for single_obs in obs:
            r = self.rng.random()
            action = None

            #select best action according to policy
            if not eps_greedy or r > self.epsilon:
                with torch.no_grad():
                    self.policy_net.eval()
                    q_values = self.policy_net(single_obs)
                    q_values = q_values.reshape(self.num_objectives, self.num_actions)
                    scalarised_values = self.scalarisation_method.scalarise_actions(q_values, self.objective_weights)
                    action = torch.argmax(scalarised_values).item()

            else: # choose random action
                action = self.rng.choice(self.num_actions)

            joint_action.append(action)

        return tuple(joint_action)

    def evaluate(self, num_repetitions: int = 5, num_points: int = 66, hv_reference_point: np.ndarray = None, seed: int = None, episode_recording_interval: int = None, render_episodes: bool = False):
        """ Evaluates the performance of the trained network by conducting num_repetitions episodes for each objective weights tuple. 
            the parameter num_points determines how many points in the objective-weight space are being explored. These weights
            are spaced equally according to the pymoo implementation: https://pymoo.org/misc/reference_directions.html.
            The recorded rewards for a specific tuple of objective weights divided by the maximum number of iterations within the episode
            to have an upper bound of 1. Each of the num_repetitions runs is returned but it is recommended to report on the average 
            to obtain a less biased result.
            The hv_reference_point is a vector specifying the best possible vectorial reward vector."""
        
        self.eval_env = self.env
        if episode_recording_interval is not None:
            self.eval_env = RecordVideoV0(self.env, video_folder="videos", name_prefix="training_MODQN", 
                                                episode_trigger=lambda x: x % episode_recording_interval == 0, fps=30)
        
        self.rng = np.random.default_rng(seed)
        #get equally spaced objective weights
        objective_weights = get_reference_directions("energy", n_dim = self.num_objectives, n_points = num_points, seed=seed)
        objective_weights = torch.from_numpy(objective_weights).to(self.device)
        
        feature_names = ["repetition_number", "weight_index","weight_tuple", "num_iterations"]
        feature_names.extend([f"normalised_{x}" for x in self.objective_names])
        feature_names.extend([f"raw_{x}" for x in self.objective_names])
        eval_logger = DataLogger("evaluation_logger",feature_names)
        
        for tuple_index in trange(objective_weights.shape[0], desc="Weight tuple", mininterval=1):#
            weight_tuple = objective_weights[tuple_index]
            self.objective_weights = weight_tuple

            # explicitly set objective weights in the environment object as well
            # so that observations are correct
            for v in self.env.unwrapped.controlled_vehicles:
                v.objective_weights = self.objective_weights

            for repetition_nr in range(num_repetitions):
                self.terminated = False
                self.truncated = False
                self.obs, _ = self.eval_env.reset()                
                accumulated_reward = np.zeros(self.num_objectives)
                curr_num_iterations = 0
                while not (self.terminated or self.truncated):
                    if render_episodes:
                        self.eval_env.render()

                    #select action based on obs. Execute action, add up reward, next iteration
                    self.obs = [torch.tensor(single_obs, device=self.device) for single_obs in self.obs] #reshape observations and
                    self.obs = [single_obs[~torch.isnan(single_obs)].reshape(1,-1) for single_obs in self.obs] #remove nan values
                    self.action = self.act(self.obs)
                    (
                    self.obs,
                    self.reward,
                    self.terminated,
                    self.truncated,
                    info,
                    ) = self.eval_env.step(self.action)
                    
                    accumulated_reward = accumulated_reward + self.reward
                    curr_num_iterations += 1

                #episode ended
                normalised_reward = accumulated_reward / curr_num_iterations
                eval_logger.add(repetition_nr, tuple_index, weight_tuple.tolist(), curr_num_iterations, *normalised_reward.tolist(), *accumulated_reward.tolist())
                
        return eval_logger.to_dataframe()

    def reduce_epsilon(self, max_iteration, eps_start, eps_end):
        self.epsilon = self.epsilon - (eps_start-eps_end)/max_iteration

    def set_objective_weights(self, weights: torch.Tensor):
        self.objective_weights = weights.to(self.device)

    def store_network_weights(self, model_path: str, model_name: str):
        torch.save(self.policy_net.state_dict(), f"{model_path}_{model_name}")

    def load_network_weights(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path))
        self.target_net.load_state_dict(torch.load(model_path))