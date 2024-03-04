"""
Rotational Equivariant ResNet and U-net
"""

import torch
from escnn import gspaces
from escnn import nn
from escnn.nn import *
from escnn.gspaces import *

def get_Net():
    pde_cnn = PDE_G_UNet2()
    return pde_cnn

##### Rotational Equivariant ResNet #####
class rot_resblock(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_dim,
                 kernel_size, N):
        super(rot_resblock, self).__init__()

        # Specify symmetry transformation
        r2_act = gspaces.rot2dOnR2(N)
        feat_type_in = nn.FieldType(r2_act, input_channels * [r2_act.regular_repr])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [r2_act.regular_repr])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.InnerBatchNorm(feat_type_hid),
            nn.ReLU(feat_type_hid)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.InnerBatchNorm(feat_type_hid),
            nn.ReLU(feat_type_hid)
        )

        self.upscale = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.InnerBatchNorm(feat_type_hid),
            nn.ReLU(feat_type_hid)
        )

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim

    def forward(self, x):
        out = self.layer1(x)
        print(x.size())
        if self.input_channels != self.hidden_dim:
            out = self.layer2(out) + self.upscale(x)
        else:
            out = self.layer2(out) + x

        return out


##### Rotational Equivariant ResNet #####
class PDE_G_UNet2(torch.nn.Module):
    def __init__(self, kernel_size = 3, N=4):
        super(PDE_G_UNet2, self).__init__()
        r2_act = gspaces.rot2dOnR2(N=4)
        # we use rho_1 representation since the input is velocity fields
        self.feat_type_in = nn.FieldType(r2_act, [r2_act.irrep(1)])
        # we use regular representation for middle layers
        self.feat_type_in_hid = nn.FieldType(r2_act, 2 * [r2_act.regular_repr])
        self.feat_type_hid_out = nn.FieldType(r2_act, 30 * [r2_act.regular_repr])
        self.feat_type_out = nn.FieldType(r2_act, 2*[r2_act.irrep(1)])

        self.input_layer = nn.SequentialModule(
            nn.R2Conv(self.feat_type_in, self.feat_type_in_hid, kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.InnerBatchNorm(self.feat_type_in_hid),
            nn.ReLU(self.feat_type_in_hid)
        )
        layers = [self.input_layer]
        layers += [rot_resblock(2, 4, kernel_size, N), rot_resblock(4, 4, kernel_size, N)]
        layers += [rot_resblock(4, 8, kernel_size, N), rot_resblock(8, 8, kernel_size, N)]
        layers += [rot_resblock(8, 16, kernel_size, N), rot_resblock(16, 16, kernel_size, N)]
        layers += [rot_resblock(16, 30, kernel_size, N), rot_resblock(30, 30, kernel_size, N)]
        layers += [nn.R2Conv(self.feat_type_hid_out, self.feat_type_out, kernel_size=kernel_size,
                             padding=(kernel_size - 1) // 2)]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, u, v):
        u = u[:, None, :, :]
        v = v[:, None, :, :]
        x_zeta = torch.cat([u, v], dim=1)

        x = nn.GeometricTensor(x_zeta, self.feat_type_in)
        x = self.model(x)

        x1 = x.tensor
        hfu = x1[:, 0, :, :]
        vfu = x1[:, 1, :, :]
        hfv = x1[:, 2, :, :]
        vfv = x1[:, 3, :, :]

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
