import torch
from escnn import gspaces
from escnn import nn
from torch.distributions import normal
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from g_unet_parts_p4m import *

def get_Net():
    pde_cnn = PDE_G_UNet2()
    return pde_cnn

class PDE_G_UNet2(torch.nn.Module):
    def __init__(self, hidden_size = 12, trilinear=True, r2_act = gspaces.rot2dOnR2(N=4)):
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

        # m = normal.Normal(0, 0.01)
        # n_u = m.sample([u.size(dim=0),128, 128])
        # n_v = m.sample([v.size(dim=0),128, 128])

        n_u = torch.empty(u.size(dim=0),128, 128).normal_(mean=0, std=0.01)
        n_v = torch.empty(v.size(dim=0), 128, 128).normal_(mean=0, std=0.01)

        # n_u = torch.randn(u.size(dim=0),128, 128) * 0.01 + 0
        # n_v = torch.randn(v.size(dim=0), 128, 128) * 0.01 + 0

        n_u = n_u[:, None, :, :].to(device)
        n_v = n_v[:, None, :, :].to(device)

        u = u[:, None, :, :]
        v = v[:, None, :, :]

        u = u + n_u
        v = v + n_v

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
        # print(x.size())
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