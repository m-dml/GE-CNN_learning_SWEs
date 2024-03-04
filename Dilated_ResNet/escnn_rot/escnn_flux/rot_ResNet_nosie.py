import torch
from escnn import gspaces
from escnn import nn
from escnn.nn import *
from escnn.gspaces import *

def get_Net():
    pde_cnn = DilatedResNet()
    return pde_cnn

class DilatedResNet(torch.nn.Module):
    def __init__(self, blocks:int=4,dilate:bool=True,features:int=12, r2_act = gspaces.rot2dOnR2(N=4)):
        super(DilatedResNet, self).__init__()
        self.r2_act = r2_act
        self.feat_type_in = nn.FieldType(r2_act, 1*[r2_act.irrep(1)])
        # self.feat_type_mid = nn.FieldType(r2_act, 48 * [r2_act.irrep(1)])
        # self.feat_type_out = nn.FieldType(r2_act, 2*[r2_act.irrep(1)])

        # self.encoderConv = nn.Conv2d(inFeatures, features, kernel_size=3, stride=1, dilation=1, padding=1)
        self.encoderConv = nn.R2Conv(nn.FieldType(r2_act, 1*[r2_act.irrep(1)]),
                                     nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                                     kernel_size=3, dilation=1, stride=1, padding=1, padding_mode="circular")



        self.blocks = torch.nn.ModuleList([])
        for _ in range(blocks):
            dils = [2,4,8,4,2] if dilate else [1,1,1,1,1]
            self.blocks.append(
                torch.nn.Sequential(
                    nn.R2Conv(nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              kernel_size=3, stride=1, dilation=1, padding=1, padding_mode="circular"),
                    nn.ReLU(nn.FieldType(r2_act, features * [r2_act.regular_repr]), inplace=True),
                    nn.R2Conv(nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              kernel_size=3, stride=1, dilation=dils[0], padding=dils[0], padding_mode="circular"),
                    nn.ReLU(nn.FieldType(r2_act, features * [r2_act.regular_repr]),inplace=True),
                    nn.R2Conv(nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              kernel_size=3, stride=1, dilation=dils[1], padding=dils[1], padding_mode="circular"),
                    nn.ReLU(nn.FieldType(r2_act, features * [r2_act.regular_repr]),inplace=True),
                    nn.R2Conv(nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              kernel_size=3, stride=1, dilation=dils[2], padding=dils[2], padding_mode="circular"),
                    nn.ReLU(nn.FieldType(r2_act, features * [r2_act.regular_repr]),inplace=True),
                    nn.R2Conv(nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              kernel_size=3, stride=1, dilation=dils[3], padding=dils[3], padding_mode="circular"),
                    nn.ReLU(nn.FieldType(r2_act, features * [r2_act.regular_repr]),inplace=True),
                    nn.R2Conv(nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              kernel_size=3, stride=1, dilation=dils[4], padding=dils[4], padding_mode="circular"),
                    nn.ReLU(nn.FieldType(r2_act, features * [r2_act.regular_repr]),inplace=True),
                    nn.R2Conv(nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                              kernel_size=3, stride=1, dilation=1, padding=1, padding_mode="circular"),
                    nn.ReLU(nn.FieldType(r2_act, features * [r2_act.regular_repr]),inplace=True),
                )
            )

        self.decoderConv = nn.R2Conv(nn.FieldType(r2_act, features * [r2_act.regular_repr]),
                                     nn.FieldType(r2_act, 2*[r2_act.irrep(1)]),
                                     kernel_size=3, stride=1, dilation=1, padding=1, padding_mode="circular")


    # def forward(self, x, time_emb=None):
    def forward(self, u, v, time_emb=None):
        u = u[:, None, :, :]
        v = v[:, None, :, :]
        x_zeta = torch.cat([u, v], dim=1)

        x = nn.GeometricTensor(x_zeta, self.feat_type_in)

        x = self.encoderConv(x)
        skipX = x

        for block in self.blocks:
            x = block(x) + skipX
            skipX = x

        x = self.decoderConv(x)
        # print(x.size())
    #################
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
