import torch
import torch.nn as nn
from g_unet_parts_p4m import *

def get_Net():
    pde_cnn = PDE_G_UNet2()
    return pde_cnn

class PDE_G_UNet2(nn.Module):
    def __init__(self, hidden_size= 16, trilinear=True):
        super(PDE_G_UNet2, self).__init__()
        self.hidden_size = hidden_size
        self.trilinear = trilinear

        self.inc = DoubleConv1(19, hidden_size)
        self.down1 = Down(hidden_size, 2 * hidden_size)
        self.down2 = Down(2 * hidden_size, 4 * hidden_size)
        self.down3 = Down(4 * hidden_size, 8 * hidden_size)
        factor = 2 if trilinear else 1
        self.down4 = Down(8 * hidden_size, 16 * hidden_size // factor)
        self.up1 = Up1(16 * hidden_size, 8 * hidden_size // factor, trilinear)
        self.up2 = Up2(8 * hidden_size, 4 * hidden_size // factor, trilinear)
        self.up3 = Up3(4 * hidden_size, 2 * hidden_size // factor, trilinear)
        self.up4 = Up4(2 * hidden_size, hidden_size, trilinear)
        self.outc = OutConv(hidden_size, 4)

    def forward(self, u, v, h):
        u = u[:, None, :, :]
        v = v[:, None, :, :]
        h = h[:, None, :, :]

        ################### add mask
        size_boundary = 1
        boundary_mask_ud = torch.ones_like(u)  # [20, 1, 128, 128]
        boundary_mask_ud[:, :, size_boundary:-size_boundary, :] = 0

        boundary_mask_lr = torch.ones_like(u)  # [20, 1, 128, 128]
        boundary_mask_lr[:, :, :, size_boundary:-size_boundary] = 0

        mask = torch.ones_like(u)
        mask[:, :, 0:size_boundary, :] = 0
        mask[:, :, :, 0:size_boundary] = 0
        mask[:, :, -size_boundary:, :] = 0
        mask[:, :, :, -size_boundary:] = 0

        boundary_mask = torch.ones_like(u) - mask

        x_zeta = torch.cat(
            [u, v, h, boundary_mask_ud, boundary_mask_lr, mask, boundary_mask,
             boundary_mask_ud * u, boundary_mask_lr * u, mask * u,
             boundary_mask * u,
             boundary_mask_ud * v, boundary_mask_lr * v, mask * v,
             boundary_mask * v,
             boundary_mask_ud * h, boundary_mask_lr * h, mask * h,
             boundary_mask * h], dim=1)

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

        # hfu_out_1 = torch.index_select(hfu_in, 3, torch.LongTensor([0]).cuda()).cuda()
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
        # hfv_out_1 = torch.index_select(hfv_in, 3, torch.LongTensor([0]))
        hfv_out = torch.cat((hfv_in[:, :, :, 1:], hfv_out_1), 3)  # flux in left

        vfv_in = vfv[:, None, :, :]

        # vfv_out_1 = torch.index_select(vfv_in, 2, torch.LongTensor([0]).cuda()).cuda()
        vfv_out_1 = vfv_in[:, :, 0, :]
        vfv_out_1 = vfv_out_1[:, :, None, :]
        # vfv_out_1 = torch.index_select(vfv_in, 2, torch.LongTensor([0]))
        vfv_out = torch.cat((vfv_in[:, :, 1:, :], vfv_out_1), 2)  # flux in left

        #### for pressure

        # out_u = 10 * torch.tanh((u + hfu_in - hfu_out + vfu_in - vfu_out) / 10)  # 10
        # out_v = 10 * torch.tanh((v + hfv_in - hfv_out + vfv_in - vfv_out) / 10)  # 10

        out_u = u + hfu_in - hfu_out + vfu_in - vfu_out
        out_v = v + hfv_in - hfv_out + vfv_in - vfv_out

        out_u = torch.squeeze(out_u)
        out_v = torch.squeeze(out_v)

        return out_u, out_v