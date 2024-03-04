import numpy as np
import torch
from g_Learning_PDE_net_mask_p import get_Net
from matplotlib.ticker import FuncFormatter
from matplotlib import rc
import matplotlib.colors as colors
import matplotlib.ticker
import math
import h5py

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


################################################
## ML model prediction
################################################

path = "/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/p4_flux/results/model_16_nl.pth"
device = torch.device('cuda')
# device = torch.device('cpu')
model = get_Net()
model.load_state_dict(torch.load(path, map_location=device))
model.eval()


iit = 0.
s_steps = 400
number_case = 20
nx = 128
my = 128
u_p4_flux = torch.zeros(number_case, s_steps, nx, my)
v_p4_flux = torch.zeros(number_case, s_steps, nx, my)
h_p4_flux = torch.zeros(number_case, s_steps, nx, my)

# u_ref = torch.zeros(number_case, s_steps, nx, my)
# v_ref = torch.zeros(number_case, s_steps, nx, my)
# h_ref = torch.zeros(number_case, s_steps, nx, my)

for iir in range(number_case):     ## iir run times
    print(iir)
    fn = iir + 1
    data_u = h5py.File('/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/generate_data/data_u_' + str(fn) + '.h5', 'r')
    data_v = h5py.File('/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/generate_data/data_v_' + str(fn) + '.h5', 'r')
    data_h = h5py.File('/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/generate_data/data_h_' + str(fn) + '.h5', 'r')

    u = data_u['data_u']  # [600, 1, 128, 128]
    uu = u[50:, 0, :, :]  # take data from 50: 600  [550, 128, 128]
    uu = torch.from_numpy(uu)

    v = data_v['data_v']  # [600, 1, 128, 128]
    vv = v[50:, 0, :, :]  # take data from 50: 600  [550, 128, 128]
    vv = torch.from_numpy(vv)

    h = data_h['data_h']  # [600, 1, 128, 128]
    hh = h[50:, 0, :, :]  # take data from 50: 600 [550, 128, 128]
    hh = torch.from_numpy(hh)

    # ################### save data
    # torch.save(uu.cpu(),
    #            '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data_prediction/ref_u' + str(fn) + '.pt')
    # torch.save(vv.cpu(),
    #            '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data_prediction/ref_v' + str(fn) + '.pt')
    # torch.save(hh.cpu(),
    #            '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data_prediction/ref_h' + str(fn) + '.pt')

    #### save reference
    # u_ref[iir, ...] = uu[0:s_steps,...]
    # v_ref[iir, ...] = vv[0:s_steps,...]
    # h_ref[iir, ...] = hh[0:s_steps,...]

    u = uu[0, ...]
    v = vv[0, ...]
    h = hh[0, ...]

    ##################################
    #           time loop            #
    ##################################
    for tn in range(s_steps):

        u = u[None, :, :]
        v = v[None, :, :]
        h = h[None, :, :]

        iit = iit + 1

        ### learing zeta
        with torch.no_grad():
            u, v = model(u.float(), v.float(), h.float())

        u = u.detach()
        v = v.detach()

        #####################################
        # calculate verticity
        #####################################
        L = 2 * math.pi
        ln = 128
        dx = L / (ln)
        vv = v
        uu = u
        # print(v.size())
        v0 = torch.index_select(vv, 1, torch.LongTensor([0]))
        v1 = torch.cat((vv, v0), 1)
        dvdx = (v1[:, 1:] - v1[:, :-1]) / dx
        ### take one move
        indices = torch.tensor([127])
        b = torch.index_select(dvdx, 1, indices)
        dvdx = torch.cat((b, dvdx[:, :-1]), 1)  # flux in left

        ##############   correct  for dudy
        u0 = torch.index_select(uu, 0, torch.LongTensor([0]))
        u1 = torch.cat((uu, u0), 0)
        dudy = (u1[1:, :] - u1[:-1, :]) / dx
        # ### take one move
        indices = torch.tensor([127])
        b = torch.index_select(dudy, 0, indices)
        dudy = torch.cat((b, dudy[:-1, :]), 0)  # flux in left

        h = dvdx - dudy

        # print(h)

        ######################################
        #  save data
        ####################################
        u_p4_flux[iir, tn, ...] = u
        v_p4_flux[iir, tn, ...] = v
        h_p4_flux[iir, tn, ...] = h

torch.save(u_p4_flux.cpu(),
           '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data/u_p4_flux.pt')
torch.save(v_p4_flux.cpu(),
           '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data/v_p4_flux.pt')
torch.save(h_p4_flux.cpu(),
           '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data/h_p4_flux.pt')


# torch.save(u_ref.cpu(),
#            '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data/u_ref.pt')
# torch.save(v_ref.cpu(),
#            '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data/v_ref.pt')
# torch.save(h_ref.cpu(),
#            '/gpfs/work/huangy1/Oceananigans.jl/examples/deep_model/3_input/diff_ICs/data/h_ref.pt')



