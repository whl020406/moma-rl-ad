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