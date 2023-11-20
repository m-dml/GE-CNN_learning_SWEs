import numpy as np
import torch
from torch.optim import Adam
from net_work import get_Net
from PDE_setups import CNN_dataset

def toCuda(x):
	if type(x) is tuple:
		return [xi.cuda() for xi in x]
	return x.cuda()
def toCpu(x):
	if type(x) is tuple:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()


# Hyper-parameters
n_epochs = 60
n_batches_per_epoch = 5000

nx, nx_u, dx, dt = 200, 199, 10.e3, 300.
cd, g, wimp = 1.e-3, 9.81, 0.5

# initialize fluid model
model = toCuda(get_Net())
model.train()

# initialize Optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# initialize Dataset
dataset = CNN_dataset(nx, nx_u, dx, dt, cd, g, wimp, batch_size=100, dataset_size=5000, average_sequence_length=5000)

def loss_function(x):
    return torch.pow(x,2)

def bandedMx(mii, up1, down1, x):
    Mx = mii*x
    Mx[:, :, :-1] += up1*x[:, :, 1:]
    Mx[:, :, 1:] += down1*x[:, :, :-1]
    return Mx

loss_vals = []

# training loop
for epoch in range(0, n_epochs):
    epoch_loss = []
    for i in range(n_batches_per_epoch):
        zeta, u, h, boundary_mask_z, zeta_mask, boundary_mask_u, u_mask, d, us = toCuda(dataset.ask())
        M, up1, down1, b = dataset.askMb()
        zeta_new = model(zeta, u, h, boundary_mask_z, zeta_mask, boundary_mask_u, u_mask)

        Mx = bandedMx(M, up1, down1, zeta_new)

        loss_linear_eq = torch.mean(loss_function(Mx - b), dim=(1, 2))

        # optional: additional loss to keep mean of zeta close to 0
        loss_mean_zeta = torch.mean(zeta_new, dim=(1, 2)) ** 2

        loss = torch.mean(25 * loss_linear_eq + 0.1 * loss_mean_zeta)

        # calculate u and h
        u, h = dataset.askuh(toCpu(zeta_new))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dataset.tell(toCpu(zeta_new), toCpu(u), toCpu(h))

        loss = toCpu(loss).numpy()
        epoch_loss.append(loss.item())

        # log training metrics
        if i % 10 == 0:
            loss_linear_eq = toCpu(torch.mean(loss_linear_eq)).numpy()
            loss_mean_zeta = toCpu(torch.mean(loss_mean_zeta)).numpy()

            if i % 100 == 0:
                print(f"{epoch}: i:{i}: loss: {loss}; loss_nav: {loss_linear_eq}; loss_boundary: {loss_mean_zeta};")

    loss_vals.append(sum(epoch_loss) / len(epoch_loss))

path = "/gpfs/work/huangy1/GE-CNN_learning_SWEs/model.pth"  # change it to your path
torch.save(model.state_dict(), path)
torch.save(np.linspace(1, n_epochs, n_epochs).astype(int), "/gpfs/work/huangy1/GE-CNN_learning_SWEs/epochs.pt")
torch.save(loss_vals, "/gpfs/work/huangy1/GE-CNN_learning_SWEs/loss_vals.pt")