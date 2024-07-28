import torch
import torch.nn as nn
class DQN_Network(nn.Module):

    def __init__(self, n_observations, n_actions, n_objectives, hidden_sizes = [128, 128]):
        super(DQN_Network, self).__init__()
        h1_input = hidden_sizes[0]
        h_last_out = hidden_sizes[-1]
        self.first_layer = nn.ModuleList([nn.Linear(n_observations, h1_input) for _ in range(n_objectives)])
        self.hidden_layers = []
        for i in range(1, len(hidden_sizes)):
            prev_size = hidden_sizes[i-1]
            layer_size = hidden_sizes[i]
            self.hidden_layers.append(nn.ModuleList([nn.Linear(prev_size, layer_size) for _ in range(n_objectives)]))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.last_layer = nn.ModuleList([nn.Linear(h_last_out, n_actions) for _ in range(n_objectives)])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        n_objectives = len(self.first_layer)
        #through first layer
        x_arr = [torch.relu(self.first_layer[i](x)) for i in range(n_objectives)]
        #through middle hidden layers
        for layer in self.hidden_layers:
            x_arr = [torch.relu(layer[i](x_arr[i])) for i in range(n_objectives)]
        #through output layer
        x_arr = [self.last_layer[i](x_arr[i]) for i in range(n_objectives)]
        stacked_arr = torch.stack(x_arr, dim=1)
        return stacked_arr
    
class Multi_DQN_Network(nn.Module):
    def __init__(self, n_observations, n_actions, n_objectives, hidden_sizes = [128,128,128]):
        super(Multi_DQN_Network, self).__init__()
        ego_net = DQN_Network(n_observations, n_actions, n_objectives, hidden_sizes)
        social_net = DQN_Network(n_observations, n_actions, n_objectives, hidden_sizes)
        self.network = nn.ModuleList([ego_net, social_net])
    
    def forward(self, x):
        output = [net(x) for net in self.network]
        stacked_output = torch.stack(output, dim=0)
        return stacked_output


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