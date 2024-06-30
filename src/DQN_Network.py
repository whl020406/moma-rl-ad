import torch
import torch.nn as nn
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
    

class Single_Objective_DQN_Network(nn.Module):

    def __init__(self, n_observations, n_actions, n_objectives):
        super(Single_Objective_DQN_Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = torch.tensor(x.flatten())
        x = torch.relu(self.layer1(x)) 
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class Separated_DQN():
    def __init__(self, n_observations, n_actions, n_objectives, loss_function, device):
        self.n_objectives = n_objectives
        self.n_actions = n_actions
        self.loss_function = loss_function()
        self.device = device
        self.networks = []
        self.optimisers = []

        for _ in range(n_objectives):
            network = Single_Objective_DQN_Network(n_observations, n_actions, 1).to(device)
            self.networks.append(network)
            self.optimisers.append(torch.optim.AdamW(network.parameters(), lr=1e-4, amsgrad=True))
        
    def get_policy_evaluation(self, observations, actions, obj_index):
        values = self.networks[obj_index](observations).reshape(observations.shape[0],self.n_actions)
        state_action_values = values.gather(1, actions)
        return state_action_values
    
    def get_next_state_values(self, next_obs, obj_index):
        with torch.no_grad():
            v = self.networks[obj_index](next_obs)
            v_max = v.max(1).values
            next_state_values = v_max
            return next_state_values
    
    def update_policy_weights(self, state_action_values, exp_state_action_values, obj_index):
        loss = self.loss_function(state_action_values, exp_state_action_values)
        self.optimisers[obj_index].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.networks[obj_index].parameters(), 100)
        self.optimisers[obj_index].step()

    def copy_network_params(self, policy_nets):
        for i in range(self.n_objectives):
            self.networks[i].load_state_dict(policy_nets[i].state_dict())

    def act(self, obs):
        q_values = torch.zeros(size=(self.n_objectives, self.n_actions)).to(self.device)
        with torch.no_grad():
            for i in range(self.n_objectives):
                q_values[i,:] = self.networks[i](obs).reshape(1,self.n_actions)
        return q_values