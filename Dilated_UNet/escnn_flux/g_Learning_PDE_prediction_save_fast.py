import numpy as np
import torch
from g_Learning_PDE_net_mask_p4m_p import get_Net
from matplotlib.ticker import FuncFormatter
from matplotlib import rc
import matplotlib.colors as colors
import matplotlib.ticker
import math

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def MyFormatter(x,lim):
    if x == 0:
        return 0
    return '{0:.1f}e{1:.0f}'.format(np.sign(x)*10**(-np.floor(np.log10(abs(x)))+np.log10(abs(x))),np.floor(np.log10(abs(x))))

majorFormatter = FuncFormatter(MyFormatter)

path = "/gpfs/work/huangy1/INS/Dilated_UNet/escnn_flux/results/model_18_1.pth"
device = torch.device('cuda')
# device = torch.device('cpu')
model = get_Net()
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

u0 = torch.load("/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/data_prediction/u0.pt")
v0 = torch.load("/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/data_prediction/v0.pt")
h0 = torch.load("/gpfs/work/huangy1/Oceananigans.jl/examples/generate_data_3/data_prediction/h0.pt")


iit = 0.
# batch_size = 1
u = u0
v = v0
h = h0

# print(type(u))
# print(u)
# print(u.size())
# p = torch.tensor(p)
##################################
#           time loop            #
##################################
for tn in range(600):
    u = u[None, :, :]
    v = v[None, :, :]
    h = h[None, :, :]

    iit = iit + 1

    ### learing zeta
    with torch.no_grad():
        u, v = model(u.float(), v.float())

    u = u.detach()
    v = v.detach()

    ######################################
    #  save data
    ####################################
    if iit == 1:
        print(iit)
        torch.save(u.cpu(), "plot/u_1.pt")
        torch.save(v.cpu(), "plot/v_1.pt")

    if iit == 30:
        print(iit)
        torch.save(u.cpu(), "plot/u_30.pt")
        torch.save(v.cpu(), "plot/v_30.pt")

    if iit == 60:
        print(iit)
        torch.save(u.cpu(), "plot/u_60.pt")
        torch.save(v.cpu(), "plot/v_60.pt")

    if iit == 90:
        print(iit)
        torch.save(u.cpu(), "plot/u_90.pt")
        torch.save(v.cpu(), "plot/v_90.pt")


    if iit == 120:
        print(iit)
        torch.save(u.cpu(), "plot/u_120.pt")
        torch.save(v.cpu(), "plot/v_120.pt")

    if iit == 150:
        print(iit)
        torch.save(u.cpu(), "plot/u_150.pt")
        torch.save(v.cpu(), "plot/v_150.pt")

    if iit == 180:
        print(iit)
        torch.save(u.cpu(), "plot/u_180.pt")
        torch.save(v.cpu(), "plot/v_180.pt")


    if iit == 210:
        print(iit)
        torch.save(u.cpu(), "plot/u_210.pt")
        torch.save(v.cpu(), "plot/v_210.pt")

    if iit == 240:
        print(iit)
        torch.save(u.cpu(), "plot/u_240.pt")
        torch.save(v.cpu(), "plot/v_240.pt")

    if iit == 270:
        print(iit)
        torch.save(u.cpu(), "plot/u_270.pt")
        torch.save(v.cpu(), "plot/v_270.pt")

    if iit == 300:
        print(iit)
        torch.save(u.cpu(), "plot/u_300.pt")
        torch.save(v.cpu(), "plot/v_300.pt")

    if iit == 330:
        print(iit)
        torch.save(u.cpu(), "plot/u_330.pt")
        torch.save(v.cpu(), "plot/v_330.pt")

    if iit == 360:
        print(iit)
        torch.save(u.cpu(), "plot/u_360.pt")
        torch.save(v.cpu(), "plot/v_360.pt")

    if iit == 390:
        print(iit)
        torch.save(u.cpu(), "plot/u_390.pt")
        torch.save(v.cpu(), "plot/v_390.pt")

    if iit == 420:
        print(iit)
        torch.save(u.cpu(), "plot/u_420.pt")
        torch.save(v.cpu(), "plot/v_420.pt")

    if iit == 450:
        print(iit)
        torch.save(u.cpu(), "plot/u_450.pt")
        torch.save(v.cpu(), "plot/v_450.pt")


