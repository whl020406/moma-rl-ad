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
from mo_gymnasium import MONormalizeReward
from copy import deepcopy
from src.utils import calc_hypervolume
import pandas as pd

class MO_DQN:
    """ 
    Implements multi-objective DQN working with one agent. Code is based on:
    https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/single_policy/ser/mo_q_learning.py#L152
    """

    def __init__(self, env: gym.Env | None, device: device = None, seed: int | None = None, 
        observation_space_shape: Sequence[int] = [1,1], num_objectives: int = 2, num_actions: int = 5, 
        replay_enabled: bool = True, replay_buffer_size: int = 1000, batch_ratio: float = 0.2, objective_weights: Sequence[float] = None,
        loss_criterion: _Loss = nn.SmoothL1Loss, gamma: float = 0.99,
        objective_names: List[str] = ["speed_reward", "energy_reward"], scalarisation_method = LinearScalarisation, scalarisation_argument_list: List = [],
        use_reward_normalisation_wrapper: bool = False, use_default_reward_normalisation: bool = True,
        network_hidden_sizes: List[int] = None, use_double_q_learning: bool = True) -> None:
        
        self.gamma = gamma

        if objective_names is None:
            objective_names = [f"reward_{x}" for x in range(num_objectives)]
        
        assert len(objective_names) == num_objectives, "The number of elements in the objective_names list must be equal to the number of objectives!"
        self.objective_names = objective_names

        self.use_reward_normalisation_wrapper = use_reward_normalisation_wrapper
        self.use_default_reward_normalisation = use_default_reward_normalisation
        self.use_double_q_learning = use_double_q_learning

        #applies reward normalisation wrapper to all objectives
        env.unwrapped.configure({"normalize_reward": use_default_reward_normalisation})
        if use_reward_normalisation_wrapper:
            for i in range(num_objectives):
                env = MONormalizeReward(env, idx=i, gamma=self.gamma)

        self.env = env
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.env.unwrapped.configure({"rng": self.rng})


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
        self.__create_network(np.cumprod(observation_space_shape)[-1], self.num_actions, self.num_objectives, network_hidden_sizes)

        self.replay_enabled = replay_enabled
        self.rb_size = replay_buffer_size
        self.batch_ratio = batch_ratio

        self.loss_criterion = loss_criterion

        #initialise replay buffer
        self.buffer = ReplayBuffer(self.rb_size, np.cumprod(observation_space_shape)[-1], self.num_objectives, self.device, self.rng, prioritise_crashes=False)

        #initialise scalarisation function
        self.scalarisation_method = scalarisation_method(*scalarisation_argument_list)


    def __create_network(self, num_observations, num_actions, num_objectives, network_hidden_sizes) -> Tuple[nn.Module, nn.Module]:
        #create one network for each objective
        if network_hidden_sizes is not None:
            policy_net = DQN_Network(num_observations, num_actions, num_objectives, network_hidden_sizes).to(self.device)
            target_net = DQN_Network(num_observations, num_actions, num_objectives, network_hidden_sizes).to(self.device)
        else:
            policy_net = DQN_Network(num_observations, num_actions, num_objectives).to(self.device)
            target_net = DQN_Network(num_observations, num_actions, num_objectives).to(self.device)

        target_net.load_state_dict(policy_net.state_dict())

        return policy_net, target_net

    def train(self, num_iterations: int = 1000, inv_optimisation_frequency: int = 1, inv_target_update_frequency: int = 20, 
                epsilon_start: float = 0.9, epsilon_end: float = 0, epsilon_end_time: float = 1, num_evaluations: int = 0, eval_seed: int = 11) :
        '''
        Runs the training procedure for num_iterations iterations. The inv_optimisation_frequency specifies 
        the number of iterations after which a weight update occurs.The inv_target_update_frequency specifies 
        the number of weight updates of the policy net, after which the target net weights are adjusted.
        Gamma is the discount factor for the rewards. Epsilon is the probability of a random action being selected during training.
        Its value is linearly reduced during the training procedure from epsilon_start to epsilon_end.
        num_evaluations specifies the number of equally spaced evaluation runs that are conducted throughout the training process.
        epsilon_end_time specifies the ratio of iterations that will have been conducted as epsilon reaches the value of epsilon_end
        '''
        #compute evaluation interval
        if num_evaluations != 0:
            eval_interval = round(num_iterations/num_evaluations)
        
        #initialise loss logger
        feature_names = ["iteration", "loss"]
        self.loss_logger = DataLogger("loss_logger",feature_names)

        #initialise hv_logger
        feature_names = ["iteration", "hypervolume"]
        hv_logger = DataLogger("hv_logger", feature_names)


        self.obs, _ = self.env.reset()
        self.obs = torch.tensor(self.obs[0].reshape(1,-1), device=self.device) #TODO: remove when going to multi-agent
        self.epsilon = epsilon_start
        self.optimiser = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-3, amsgrad=True)

        self.loss_func = self.loss_criterion(reduction="mean")
        episode_nr = 0
        num_of_conducted_optimisation_steps = 0
        max_eps_iteration = round(num_iterations * epsilon_end_time)
        #take step in environment
        for i in trange(num_iterations, desc="Training iterations", mininterval=2):
            self.action = self.act(self.obs, eps_greedy=True)
            self.reduce_epsilon(max_eps_iteration, epsilon_start, epsilon_end) #linearly reduce the value of epsilon
            (
                self.next_obs,
                self.reward,
                self.terminated,
                self.truncated,
                info,
            ) = self.env.step(self.action)

            #cast return values to gpu tensor before storing them in replay buffer
            self.next_obs = torch.tensor(self.next_obs[0].reshape(1,-1), device=self.device) #TODO: remove when going to multi-agent
            if self.num_objectives == 1:
                self.reward = [self.reward]
            self.reward = torch.tensor(self.reward, device=self.device)
            self.action = torch.tensor([self.action], device=self.device)
            self.terminated = torch.tensor([self.terminated], device=self.device)
            #push to replay buffer
            self.buffer.push(self.obs, self.action, self.next_obs, self.reward, self.terminated)
            self.obs = self.next_obs #use next_obs as obs during the next iteration
            
            #update the weights every optimisation_frequency steps
            if (i % inv_optimisation_frequency) == 0:
                if self.buffer.num_elements == self.rb_size:
                    self.__update_weights(i, num_of_conducted_optimisation_steps, inv_target_update_frequency)
                    num_of_conducted_optimisation_steps += 1

            #run evaluation
            if (num_evaluations != 0) and (i % eval_interval == 0):
                _, hv = self.evaluate(num_repetitions= 5, num_points= 10, hv_reference_point=np.array([0,0]),
                                        seed = eval_seed)
                hv_logger.add(iteration=i, hypervolume=hv)

            if self.terminated or self.truncated:
                episode_nr += 1
                self.obs, _ = self.env.reset()
                self.obs = torch.tensor(self.obs[0].reshape(1,-1), device=self.device) #TODO: remove when going to multi-agent
                self.objective_weights = random_objective_weights(self.num_objectives, self.rng, self.device)
        
        #prepare logger data
        df = self.loss_logger.to_dataframe()
        leading_nans = pd.DataFrame(data = np.full(shape=(df["iteration"].min(), len(df.columns)), fill_value=np.nan),columns=df.columns)
        df = pd.concat([leading_nans, df], ignore_index=True)
        #add hypervolume information if applicable
        if num_evaluations != 0:
            hv_df = hv_logger.to_dataframe()
            df["hypervolume"] = np.nan
            df.loc[df.index.isin(hv_df["iteration"]),"hypervolume"] = hv_df["hypervolume"].to_numpy()

        return df

    def __update_weights(self, current_iteration, current_optimisation_iteration, inv_target_update_frequency):
        self.policy_net.train()
        #update normal network each time the function is called
        #update target network every k steps

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
        
        with torch.no_grad():
            #code taken from https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/deep_q_network/pytorch.py
            if self.use_double_q_learning:
                best_actions_policy_net = self.policy_net(next_obs).argmax(2).unsqueeze(2)
                target_net_estimate = self.target_net(next_obs)
                next_state_values = target_net_estimate.gather(2, best_actions_policy_net).squeeze(2)
            else:
                next_state_values = self.target_net(next_obs).max(2).values

        next_state_values[term_flags] = 0

        exp_state_action_values = next_state_values * self.gamma + rewards
        #compute loss between estimates and actual values
        loss = self.loss_func(state_action_values, exp_state_action_values)
        #add loss to logger
        self.loss_logger.add(iteration=current_iteration, loss=loss.item())
        #backpropagate loss
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimiser.step()

        #update the target networks
        if (current_optimisation_iteration % inv_target_update_frequency) == 0:
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
                q_values = q_values.reshape(self.num_objectives, self.num_actions)
                scalarised_values = self.scalarisation_method.scalarise_actions(q_values, self.objective_weights)
                action = torch.argmax(scalarised_values).item()

        else: # choose random action
            action = self.rng.choice(self.num_actions)

        return action
    
    def evaluate(self, num_repetitions: int = 5, num_points: int = 66, hv_reference_point: np.ndarray = None, seed: int = None, episode_recording_interval: int = None, render_episodes: bool = False, video_name_prefix = None, video_location = None):
        """ Evaluates the performance of the trained network by conducting num_repetitions episodes for each objective weights tuple. 
            the parameter num_points determines how many points in the objective-weight space are being explored. These weights
            are spaced equally according to the pymoo implementation: https://pymoo.org/misc/reference_directions.html.
            The recorded rewards for a specific tuple of objective weights divided by the maximum number of iterations within the episode
            to have an upper bound of 1. Each of the num_repetitions runs is returned but it is recommended to report on the average 
            to obtain a less biased result.
            The hv_reference_point is a vector specifying the best possible vectorial reward vector."""
        
        self.eval_env = deepcopy(self.env) #TODO: test whether deepcopy works
        self.eval_env.unwrapped.configure({"rng": self.rng})

        if episode_recording_interval is not None:
            self.eval_env = RecordVideoV0(self.env, video_folder="videos", name_prefix="training_MODQN", 
                                                episode_trigger=lambda x: x % episode_recording_interval == 0, fps=30)
        
        self.rng = np.random.default_rng(seed)
        #get equally spaced objective weights
        objective_weights = get_reference_directions("energy", n_dim = self.num_objectives, n_points = num_points, seed=seed)
        objective_weights = torch.from_numpy(objective_weights).to(self.device)
        
        #instantiate data loggers
        #for summary information
        feature_names = ["repetition_number", "weight_index","weight_tuple", "num_iterations"]
        feature_names.extend([f"normalised_{x}" for x in self.objective_names])
        feature_names.extend([f"raw_{x}" for x in self.objective_names])
        eval_logger = DataLogger("evaluation_logger",feature_names)

        #for more detailed information on the individual vehicles
        #target and actual speeds are only useful for uncontrolled vehicles, while weights are only applicable to controlled vehicles
        feature_names = ["repetition_number", "weight_index", "weight_tuple", "iteration", "vehicle_id", "controlled_flag", "action", "target_speed", "curr_speed", "acc", "lane"]
        feature_names.extend([f"curr_{x}" for x in self.objective_names])
        vehicle_logger = DataLogger("vehicle_logger", feature_names)
        
        for tuple_index in trange(objective_weights.shape[0], desc="Weight tuple", mininterval=1):
            weight_tuple = objective_weights[tuple_index]
            self.objective_weights = weight_tuple
            
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
                    self.obs = torch.tensor(self.obs[0].reshape(1,-1), device=self.device) #TODO: remove when going to multi-agent
                    self.action = self.act(self.obs)
                    (
                    self.obs,
                    self.reward, #use that if no reward normalisation wrapper was applied
                    self.terminated,
                    self.truncated,
                    info,
                    ) = self.eval_env.step(self.action)
                    
                    #if reward normalisation wrapper is utilised or default normalisation wasn't applied,
                    #take default normalised rewards from info dict
                    if self.use_reward_normalisation_wrapper or (not self.use_default_reward_normalisation):
                        self.reward = info["rewards"]

                    #populate vehicle logger
                    for v_id, vehicle in enumerate(self.eval_env.unwrapped.road.vehicles):
                        action = np.nan
                        lane = vehicle.lane_index[2]
                        acc = vehicle.action["acceleration"]
                        reward = np.full(self.num_objectives, fill_value=np.nan)
                        if vehicle.is_controlled:
                            action = self.action
                            reward = self.reward
                        vehicle_logger.add(repetition_nr, tuple_index, weight_tuple.tolist(), curr_num_iterations, 
                                           v_id, vehicle.is_controlled, action, vehicle.target_speed, vehicle.speed, 
                                           acc, lane, *reward.tolist())
                        
                    accumulated_reward = accumulated_reward + self.reward
                    curr_num_iterations += 1

                #episode ended
                normalised_reward = accumulated_reward / curr_num_iterations
                eval_logger.add(repetition_nr, tuple_index, weight_tuple.tolist(), curr_num_iterations, *normalised_reward.tolist(), *accumulated_reward.tolist())
        
        #compute hypervolume if reference point is given
        if hv_reference_point is not None:
            df = eval_logger.to_dataframe()
            mean_df = df.groupby("weight_index")[["normalised_speed_reward", "normalised_energy_reward"]].mean()
            reward_vector = mean_df.to_numpy()
            hypervolume = calc_hypervolume(hv_reference_point, reward_vector)
            return df, vehicle_logger.to_dataframe(), hypervolume
        
        #otherwise only return the eval logger dataframe
        return eval_logger.to_dataframe(), vehicle_logger.to_dataframe()

    def reduce_epsilon(self, max_iteration, eps_start, eps_end):
        self.epsilon = max(eps_end,self.epsilon - (eps_start-eps_end)/max_iteration)

    def set_objective_weights(self, weights: torch.Tensor):
        self.objective_weights = weights.to(self.device)

    def store_network_weights(self, model_path: str, model_name: str):
        torch.save(self.policy_net.state_dict(), f"{model_path}_{model_name}")
    
    def store_network(self, model_path: str, model_name: str):
        torch.save(self.policy_net, f"{model_path}{model_name}")

    def load_network_weights(self, model_path: str):
        self.policy_net.load_state_dict(torch.load(model_path))
        self.target_net.load_state_dict(torch.load(model_path))