# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .activations import ACTIVATION_REGISTRY
from .fourier import SpectralConv2d

from escnn import gspaces
from g_unet_parts_p4m import *

def get_Net():
    pde_cnn = PDE_G_UNet2()
    return pde_cnn

class PDE_G_UNet2(torch.nn.Module):
    def __init__(self, hidden_size = 31, trilinear=True, r2_act = gspaces.rot2dOnR2(N=4)):
        super(PDE_G_UNet2, self).__init__()
        self.hidden_size = hidden_size
        self.trilinear = trilinear
        self.r2_act = r2_act

        ####################################################
        ####################################################
        self.feat_type_in = nn.FieldType(r2_act, [r2_act.irrep(1)])

        self.inc = DoubleConv1(nn.FieldType(r2_act, [r2_act.irrep(1)]),
                               nn.FieldType(r2_act, hidden_size * [r2_act.regular_repr]))
        self.down1 = Down(nn.FieldType(r2_act, hidden_size * [r2_act.regular_repr]),
                          nn.FieldType(r2_act, 2*hidden_size * [r2_act.regular_repr]))
        self.down2 = Down(nn.FieldType(r2_act, 2*hidden_size * [r2_act.regular_repr]),
                          nn.FieldType(r2_act, 4*hidden_size * [r2_act.regular_repr]))
        self.down3 = Down(nn.FieldType(r2_act, 4*hidden_size * [r2_act.regular_repr]),
                          nn.FieldType(r2_act, 8*hidden_size * [r2_act.regular_repr]))
        factor = 2
        self.down4 = Down(nn.FieldType(r2_act, 8*hidden_size * [r2_act.regular_repr]),
                          nn.FieldType(r2_act, 8*hidden_size * [r2_act.regular_repr]))

        self.up1 = Up1(nn.FieldType(r2_act, 16*hidden_size * [r2_act.regular_repr]),   # 128
                       nn.FieldType(r2_act, 8*hidden_size * [r2_act.regular_repr]),    # 64
                       nn.FieldType(r2_act, 4*hidden_size * [r2_act.regular_repr]))    # 32

        self.up2 = Up2(nn.FieldType(r2_act, 8*hidden_size * [r2_act.regular_repr]),    # 64
                       nn.FieldType(r2_act, 4*hidden_size * [r2_act.regular_repr]),    # 32
                       nn.FieldType(r2_act, 2*hidden_size * [r2_act.regular_repr]))    # 16


        self.up3 = Up3(nn.FieldType(r2_act, 4*hidden_size * [r2_act.regular_repr]),    # 32
                       nn.FieldType(r2_act, 2*hidden_size * [r2_act.regular_repr]),    # 16
                       nn.FieldType(r2_act, hidden_size * [r2_act.regular_repr]))      #  8


        self.up4 = Up4(nn.FieldType(r2_act, 2*hidden_size * [r2_act.regular_repr]),   # 16
                       nn.FieldType(r2_act, hidden_size * [r2_act.regular_repr]),     #  8
                       nn.FieldType(r2_act, hidden_size * [r2_act.regular_repr]))     #   8

        self.outc = OutConv(nn.FieldType(r2_act, hidden_size * [r2_act.regular_repr]),
                            nn.FieldType(r2_act, 2*[r2_act.irrep(1)]))

    def forward(self, u, v):
        u = u[:, None, :, :]
        v = v[:, None, :, :]

        x_zeta = torch.cat([u, v], dim=1)

        x_zeta = self.feat_type_in(x_zeta)
        x1 = self.inc(x_zeta)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        hfu, vfu, hfv, vfv = self.outc(x)

        ##### for u
        hfu_in = hfu[:, None, :, :]

        hfu_out_1 = hfu_in[:, :, :, 0]
        hfu_out_1 = hfu_out_1[:, :, :, None]
        hfu_out = torch.cat((hfu_in[:, :, :, 1:], hfu_out_1), 3)  # flux in left

        vfu_in = vfu[:, None, :, :]

        # vfu_out_1 = torch.index_select(vfu_in, 2, torch.LongTensor([0]).cuda()).cuda()
        vfu_out_1 = vfu_in[:, :, 0, :]
        vfu_out_1 = vfu_out_1[:, :, None, :]
        vfu_out = torch.cat((vfu_in[:, :, 1:, :], vfu_out_1), 2)  # flux in left

        ##### for v
        hfv_in = hfv[:, None, :, :]

        # hfv_out_1 = torch.index_select(hfv_in, 3, torch.LongTensor([0]).cuda()).cuda()
        hfv_out_1 = hfv_in[:, :, :, 0]
        hfv_out_1 = hfv_out_1[:, :, :, None]
        hfv_out = torch.cat((hfv_in[:, :, :, 1:], hfv_out_1), 3)  # flux in left

        vfv_in = vfv[:, None, :, :]

        vfv_out_1 = vfv_in[:, :, 0, :]
        vfv_out_1 = vfv_out_1[:, :, None, :]
        vfv_out = torch.cat((vfv_in[:, :, 1:, :], vfv_out_1), 2)  # flux in left

        #### for pressure
        out_u = u + hfu_in - hfu_out + vfu_in - vfu_out
        out_v = v + hfv_in - hfv_out + vfv_in - vfv_out

        out_u = torch.squeeze(out_u)
        out_v = torch.squeeze(out_v)

        return out_u, out_v

















#######################################################################
#######################################################################
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.bn1 = nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups, num_channels=planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes) if norm else nn.Identity(),
            )

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.activation(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = out + self.shortcut(x)
        # out = self.activation(out)
        return out


class DilatedBasicBlock(nn.Module):
    """Basic block for Dilated ResNet

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to True.
        num_groups (int, optional): number of groups for group normalization. Defaults to 1.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers):
            out = self.activation(layer(norm(out)))
        return out + x


class ResNet(nn.Module):
    """Class to support ResNet like feedforward architectures

    Args:
        n_input_scalar_components (int): Number of input scalar components in the model
        n_input_vector_components (int): Number of input vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        block (Callable): BasicBlock or DilatedBasicBlock or FourierBasicBlock
        num_blocks (List[int]): Number of blocks in each layer
        time_history (int): Number of time steps to use in the input
        time_future (int): Number of time steps to predict in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
    """

    padding = 9

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        block,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = True,
        diffmode: bool = False,
        usegrid: bool = False,
    ):
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels
        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        if self.usegrid:
            insize += 2
        self.conv_in1 = nn.Conv2d(
            insize,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.in_planes,
            time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2),
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    num_blocks[i],
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def _make_layer(
        self,
        block: Callable,
        planes: int,
        num_blocks: int,
        stride: int,
        activation: str,
        norm: bool = True,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        # prev = x.float()
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = layer(x)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        if self.diffmode:
            raise NotImplementedError("diffmode")
            # x = x + prev[:, -1:, ...].detach()
        return x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )