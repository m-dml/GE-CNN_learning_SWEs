import torch
import torch.distributions as tdist
from torch.distributions import normal

# m = normal.Normal(0, 0.01)
# # s = m.sample()
# s = m.sample([20, 128, 128])

s = torch.empty(20,128, 128).normal_(mean=0, std=0.01)

print(s)