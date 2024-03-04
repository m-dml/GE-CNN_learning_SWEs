import torch
from escnn import gspaces
import math

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
        self.feat_type_in = nn.FieldType(r2_act, [r2_act.irrep(1)])      # input is vector field

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


        # self.outc = OutConv(nn.FieldType(r2_act, hidden_size * [r2_act.regular_repr]),
        #                     nn.FieldType(r2_act, 4*[r2_act.regular_repr]))
        self.outc = OutConv(nn.FieldType(r2_act, hidden_size * [r2_act.regular_repr]),
                            nn.FieldType(r2_act, 1*[r2_act.trivial_repr]))    # output is a scaler field

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

        aa = self.outc(x)

        ###############################################
        L = 2 * math.pi
        ln = 128
        dx = L / ln

        ### - da/dx  for v
        vs = aa[:, :, 0]
        v0 = vs[:, :, None]
        v1 = torch.cat((aa, v0), 2)
        out_v = - (v1[:, :, 1:] - v1[:, :, :-1]) / dx

        ###  da/dy  for u
        us = aa[:, 0, :]
        u0 = us[:, None, :]
        u1 = torch.cat((aa, u0), 1)
        out_u = (u1[:, 1:, :] - u1[:, :-1, :]) / dx

        #####################################################
        out_u = out_u[:, None, :, :]
        out_v = out_v[:, None, :, :]

        # out_u = u + out_u
        # out_v = v + out_v

        out_u = torch.squeeze(out_u)
        out_v = torch.squeeze(out_v)

        return out_u, out_v