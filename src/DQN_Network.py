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