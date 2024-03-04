import torch


x = torch.rand(2, 3, 3)
print(x)
noise = torch.randn_like(x)
print(noise)