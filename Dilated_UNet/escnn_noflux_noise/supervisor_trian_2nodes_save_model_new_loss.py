"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

from torch.optim import Adam
from g_Learning_PDE_net_mask_p4m import get_Net
from timeit import default_timer
from utilities3 import *
import lightning as L
import math

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True


torch.manual_seed(0)
np.random.seed(0)

fabric = L.Fabric(accelerator="gpu", devices="auto", strategy="ddp", num_nodes=2)   # num_nodes=2, devices is 2gpu per nodes, eg. 2 gpu on g022
# fabric = L.Fabric(accelerator="gpu", devices="auto", strategy="ddp")  # old
fabric.launch()
################################################################
# configs
################################################################
# TRAIN_X_PATH = '/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/train_x_1.mat'
# TRAIN_Y_PATH = '/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/train_y_1.mat'
#
# TEST_X_PATH = '/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/test_x_1.mat'
# TEST_Y_PATH = '/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/test_y_1.mat'
#
# VALIDATE_X_PATH = '/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/validate_x_1.mat'
# VALIDATE_Y_PATH = '/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/validate_y_1.mat'

TRAIN_X_PATH = '/p/project/hai_ml_pde/generate_data_3/train_x_1.mat'
TRAIN_Y_PATH = '/p/project/hai_ml_pde/generate_data_3/train_y_1.mat'

TEST_X_PATH = '/p/project/hai_ml_pde/generate_data_3/test_x_1.mat'
TEST_Y_PATH = '/p/project/hai_ml_pde/generate_data_3/test_y_1.mat'

VALIDATE_X_PATH = '/p/project/hai_ml_pde/generate_data_3/validate_x_1.mat'
VALIDATE_Y_PATH = '/p/project/hai_ml_pde/generate_data_3/validate_y_1.mat'

ntrain = 10980*2   # 4:49410
ntest  = 2196   # 4:549*5
nvalidate = 2196

batch_size = 30   # 20

epochs = 500   # 1000
step_size = 100
gamma = 0.5

################################################################
# load data
################################################################
reader = MatReader(TRAIN_X_PATH)
train_x_u = reader.read_field('u_sol_x')
train_x_v = reader.read_field('v_sol_x')
# train_x_h = reader.read_field('h_sol_x')

reader = MatReader(TRAIN_Y_PATH)
train_y_u = reader.read_field('u_sol_y')
train_y_v = reader.read_field('v_sol_y')
# train_y_h = reader.read_field('h_sol_y')

#######################

reader = MatReader(TEST_X_PATH)
test_x_u = reader.read_field('u_sol_x')
test_x_v = reader.read_field('v_sol_x')
# test_x_h = reader.read_field('h_sol_x')

reader = MatReader(TEST_Y_PATH)
test_y_u = reader.read_field('u_sol_y')
test_y_v = reader.read_field('v_sol_y')
# test_y_h = reader.read_field('h_sol_y')

#######################
reader = MatReader(VALIDATE_X_PATH)
validate_x_u = reader.read_field('u_sol_x')
validate_x_v = reader.read_field('v_sol_x')
# validate_x_h = reader.read_field('h_sol_x')

reader = MatReader(VALIDATE_Y_PATH)
validate_y_u = reader.read_field('u_sol_y')
validate_y_v = reader.read_field('v_sol_y')
# validate_y_h = reader.read_field('h_sol_y')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x_u, train_x_v,
                                                                          train_y_u, train_y_v), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x_u,  test_x_v,
                                                                         test_y_u, test_y_v), batch_size=batch_size, shuffle=False)
validate_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(validate_x_u,  validate_x_v,
                                                                             validate_y_u, validate_y_v), batch_size=batch_size, shuffle=False)


################################################################
# training and evaluation
################################################################
model = get_Net().cuda()

print(count_params(model))

optimizer = Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

model, optimizer = fabric.setup(model, optimizer)
train_loader, validate_loader, test_loader = fabric.setup_dataloaders(train_loader, validate_loader, test_loader)

# myloss = LpLoss(size_average=False)
def loss_function(x):
    return torch.pow(x, 2)

loss_train_f = []
loss_test_f = []
loss_validate_f = []

early_stopping = EarlyStopping(tolerance=20, min_delta=10)

for ep in range(epochs):

    epoch_loss = []
    epoch_loss_test = []

    model.train()
    t1 = default_timer()
    train_l2 = 0

    for x_u, x_v, y_u, y_v in train_loader:

        optimizer.zero_grad()
        out_u, out_v = model(x_u, x_v)

        loss_trian = 1000 * torch.mean(loss_function(out_u - y_u) + loss_function(out_v - y_v), dim=(0, 1, 2))

        # loss.backward()
        fabric.backward(loss_trian)

        optimizer.step()
        train_l2 += loss_trian.item()
    scheduler.step()

    ##############################################
    ##############################################
    validate_l2 = 0.0
    model.eval()
    with torch.no_grad():
        for x_u, x_v, y_u, y_v in validate_loader:
            out_u, out_v = model(x_u, x_v )

            loss_validate = 1000 * torch.mean(loss_function(out_u - y_u) + loss_function(out_v - y_v),
                                       dim=(0, 1, 2))
            validate_l2 += loss_validate.item()

    ##################################################
    # EarlyStopping
    early_stopping(loss_trian.item(), loss_validate.item())
    if early_stopping.early_stop:
        print("We are at epoch:", ep)
        break


    # model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x_u, x_v, y_u, y_v in test_loader:
            out_u, out_v = model(x_u, x_v)

            loss_test = 1000 * torch.mean(loss_function(out_u - y_u) + loss_function(out_v - y_v), dim=(0, 1, 2))

            test_l2 += loss_test.item()

    train_l2/= ntrain
    validate_l2 /= nvalidate
    test_l2 /= ntest

    t2 = default_timer()

    loss_train_f.append(train_l2)
    loss_validate_f.append(validate_l2)
    loss_test_f.append(test_l2)

    print(ep, t2-t1, train_l2, validate_l2)



# state_dict = {
#     "model": model,
#     "optimizer" : optimizer,
#     "scheduler": scheduler
# }
# fabric.save("/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/save_result_p4/checkpoint.ckpt", state_dict)

mi= 2
path = '/p/project/hai_ml_pde/INS/Dilated_UNet/escnn_noflux_noise/results/model_12_'+ str(mi) + '.pth'
torch.save(model.state_dict(), path)
torch.save(optimizer.state_dict(), '/p/project/hai_ml_pde/INS/Dilated_UNet/escnn_noflux_noise/results/optimizer_12_'+ str(mi) + '.pt')
torch.save(scheduler.state_dict(), '/p/project/hai_ml_pde/INS/Dilated_UNet/escnn_noflux_noise/results/scheduler_12_'+ str(mi) + '.pt')


## save train loss
torch.save(np.linspace(1, epochs, epochs).astype(int), '/p/project/hai_ml_pde/INS/Dilated_UNet/escnn_noflux_noise/results/epochs_12_'+ str(mi) + '.pt')
torch.save(loss_train_f, '/p/project/hai_ml_pde/INS/Dilated_UNet/escnn_noflux_noise/results/loss_trian_f_12_'+ str(mi) + '.pt')
torch.save(loss_validate_f, '/p/project/hai_ml_pde/INS/Dilated_UNet/escnn_noflux_noise/results/loss_validate_f_12_'+ str(mi) + '.pt')
torch.save(loss_test_f, '/p/project/hai_ml_pde/INS/Dilated_UNet/escnn_noflux_noise/results/loss_test_f_12_'+ str(mi) + '.pt')



# mi= 1
# path = '/gpfs/work/huangy1/INS/escnn_fliprot/escnn_flux/results/model_16_'+ str(mi) + '.pth'
# torch.save(model.state_dict(), path)
# torch.save(optimizer.state_dict(), '/gpfs/work/huangy1/INS/escnn_fliprot/escnn_flux/results/optimizer_16_'+ str(mi) + '.pt')
# torch.save(scheduler.state_dict(), '/gpfs/work/huangy1/INS/escnn_fliprot/escnn_flux/results/scheduler_16_'+ str(mi) + '.pt')
#
# ## save train loss
# torch.save(np.linspace(1, epochs, epochs).astype(int), '/gpfs/work/huangy1/INS/escnn_fliprot/escnn_flux/results/epochs_16_'+ str(mi) + '.pt')
# torch.save(loss_train_f, '/gpfs/work/huangy1/INS/escnn_fliprot/escnn_flux/results/loss_trian_f_16_'+ str(mi) + '.pt')
# torch.save(loss_validate_f, '/gpfs/work/huangy1/INS/escnn_fliprot/escnn_flux/results/loss_validate_f_16_'+ str(mi) + '.pt')
# torch.save(loss_test_f, '/gpfs/work/huangy1/INS/escnn_fliprot/escnn_flux/results/loss_test_f_16_'+ str(mi) + '.pt')