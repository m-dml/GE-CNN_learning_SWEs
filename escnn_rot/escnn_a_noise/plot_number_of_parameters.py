import numpy as np
import torch
from g_Learning_PDE_net_mask_p4m import get_Net
from prettytable import PrettyTable

net = get_Net()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


count_parameters(net)