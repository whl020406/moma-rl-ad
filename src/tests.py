from utils import ReplayBuffer
import torch
import numpy as np

obs = torch.arange(0,40).reshape(10,4)
next_obs = torch.arange(40,80).reshape(10,4)
actions = torch.arange(80,90)
rewards = (torch.arange(1,21) * (-1)).reshape(10,2)
terminated = torch.zeros(10)

repl_buffer = ReplayBuffer(200,observation_space_shape=4, num_objectives=2, rng = np.random.default_rng(None), importance_sampling=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

for i in range(obs.shape[0]):
    repl_buffer.push(obs[i,:],torch.tensor([actions[i]]),next_obs[i,:], rewards[i,:], torch.tensor([0]), torch.tensor([i+1]))
    print("---------------------------------------")
    samples = repl_buffer.sample(10)
    ids = repl_buffer.get_importance_sampling_id(samples)
    unique_ids, counts = np.unique(ids.cpu().numpy(), return_counts=True)
    print(np.vstack([unique_ids, counts]))

for _ in range(19):
    for i in range(obs.shape[0]):
        repl_buffer.push(obs[i,:],torch.tensor([actions[i]]),next_obs[i,:], rewards[i,:], torch.tensor([0]), torch.tensor([i+1]))


print("---------------------------------------")
print("---------------------------------------")

samples = repl_buffer.sample(100)
ids = repl_buffer.get_importance_sampling_id(samples)
unique_ids, counts = np.unique(ids.cpu().numpy(), return_counts=True)
print(np.vstack([unique_ids, counts]))
