import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import rc
import matplotlib.colors as colors
import matplotlib.ticker
import types

#########################################
nx = 128
my = 128
x = np.arange(0, nx + 1, 1)
y = np.arange(0, my + 1, 1)
X_u, Y_u = np.meshgrid(y, x)

i_number = [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450]
#########################################
# fig, axs = plt.subplots(9, 8)
fig, axs = plt.subplots(nrows=7, ncols=8+8,  gridspec_kw={'hspace': 0.05}, figsize=(9, 9))  # rate of image L:W


for i in range(8+8):

    # reference simulation
    uu = torch.load(
        "/home/yunfeihuang/Documents/Oceananigans.jl/examples/generate_data_3/data_prediction/u.pt")
    ref_u = uu[i_number[i], ...]
    vminn = torch.min(ref_u)
    vmaxx = torch.max(ref_u)
    axs[0, i].pcolor(X_u, Y_u, ref_u, cmap='twilight', vmin=vminn, vmax=vmaxx)

    axs[2, 0].set_title('ref', rotation=90, y=2.8, x=-0.1, pad=-14, color="black")

    axs[0, i].set_aspect('equal', adjustable='box')
    axs[0, i].get_xaxis().set_visible(False)
    axs[0, i].get_yaxis().set_visible(False)
    axs[0, i].set_title('t=' + str(i_number[i]*0.2)+'s')
    plt.ylabel("reference")


    ##########################
    ## escnn_noflux
    escnn_flux = torch.load(
        '/home/yunfeihuang/Documents/INS/Dilated_UNet/escnn_flux/plot/plot/u_'
        + str(i_number[i]) + '.pt')

    vminn = torch.min(escnn_flux)
    vmaxx = torch.max(escnn_flux)
    axs[1, i].pcolor(X_u, Y_u, escnn_flux, cmap='twilight', vmin=vminn, vmax=vmaxx)
    axs[1, i].set_aspect('equal', adjustable='box')
    axs[1, i].get_xaxis().set_visible(False)
    axs[1, i].get_yaxis().set_visible(False)

    axs[1, 0].set_title('aa', rotation=90, y=0.1, x=-0.1, pad=-14, color="purple")

    # error
    axs[2, i].pcolor(X_u, Y_u, escnn_flux - ref_u, cmap='bwr', vmin=-0.1, vmax=0.1)
    axs[2, i].set_aspect('equal', adjustable='box')
    axs[2, i].get_xaxis().set_visible(False)
    axs[2, i].get_yaxis().set_visible(False)

    ##########################
    ## escnn_a
    escnn_a = torch.load(
        '/home/yunfeihuang/Documents/INS/Dilated_UNet/escnn_flux_noise/plot/plot/u_'
        + str(i_number[i]) + '.pt')
    vminn = torch.min(escnn_a)
    vmaxx = torch.max(escnn_a)
    axs[3, i].pcolor(X_u, Y_u, escnn_a, cmap='twilight', vmin=vminn, vmax=vmaxx)
    axs[3, i].set_aspect('equal', adjustable='box')
    axs[3, i].get_xaxis().set_visible(False)
    axs[3, i].get_yaxis().set_visible(False)

    axs[3, 0].set_title('rot_a', rotation=90, y=-0.2, x=-0.1, pad=-14, color="blue")

    # error
    axs[4, i].pcolor(X_u, Y_u, escnn_a - ref_u, cmap='bwr', vmin=-0.1, vmax=0.1)
    axs[4, i].set_aspect('equal', adjustable='box')
    axs[4, i].get_xaxis().set_visible(False)
    axs[4, i].get_yaxis().set_visible(False)

    ##########################
    ## escnn_flux
    escnn_noflux = torch.load(
        '/home/yunfeihuang/Documents/INS/Dilated_UNet/escnn_flux_noise/plot/plot/u_'
        + str(i_number[i]) + '.pt')

    vminn = torch.min(escnn_noflux)
    vmaxx = torch.max(escnn_noflux)
    axs[5, i].pcolor(X_u, Y_u, escnn_noflux, cmap='twilight', vmin=vminn, vmax=vmaxx)
    axs[5, i].set_aspect('equal', adjustable='box')
    axs[5, i].get_xaxis().set_visible(False)
    axs[5, i].get_yaxis().set_visible(False)

    axs[5, 0].set_title('rot_flux', rotation=90, y=0.2, x=-0.1, pad=-14, color="green")

    # error
    axs[6, i].pcolor(X_u, Y_u, escnn_noflux - ref_u, cmap='bwr', vmin=-0.1, vmax=0.1)
    axs[6, i].set_aspect('equal', adjustable='box')
    axs[6, i].get_xaxis().set_visible(False)
    axs[6, i].get_yaxis().set_visible(False)

plt.show()