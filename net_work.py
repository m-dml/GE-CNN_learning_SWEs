from ge_unet_parts import *

def get_Net():
    pde_cnn = PDE_GE_UNet()
    return pde_cnn

class PDE_GE_UNet(pl.LightningModule):
    def __init__(self, hidden_size=30, linear=True):
        super(PDE_GE_UNet, self).__init__()
        self.hidden_size = hidden_size
        self.linear = linear

        self.inc_z = Conv_z(8, hidden_size)
        self.inc_u = Conv_u(5, hidden_size)
        self.bn = nn.BatchNorm2d(hidden_size)
        self.relu = nn.ReLU()

        self.inc_zu = Conv_zu(hidden_size, hidden_size)

        self.down1 = Down(hidden_size, 2 * hidden_size)
        self.down2 = Down(2 * hidden_size, 4 * hidden_size)
        self.down3 = Down(4 * hidden_size, 8 * hidden_size)
        factor = 2 if linear else 1
        self.down4 = Down(8 * hidden_size, 16 * hidden_size // factor)
        self.up1 = Up1(16 * hidden_size, 8 * hidden_size // factor, linear)
        self.up2 = Up2(8 * hidden_size, 4 * hidden_size // factor, linear)
        self.up3 = Up3(4 * hidden_size, 2 * hidden_size // factor, linear)
        self.up4 = Up4(2 * hidden_size, hidden_size, linear)
        self.outc = OutConv(hidden_size, 1)

    def forward(self, zeta, u, h, boundary_mask_z, zeta_mask, boundary_mask_u, u_mask):
        x_zeta = torch.cat([zeta, h, boundary_mask_z, zeta_mask, boundary_mask_z * zeta, zeta_mask * zeta, boundary_mask_z * h, zeta_mask * h], dim=1)
        x_u = torch.cat([u, boundary_mask_u, u_mask, boundary_mask_u * u, u_mask * u], dim=1)

        x1_z = self.inc_z(x_zeta)
        x1_u = self.inc_u(x_u)
        x1_zu = x1_z + x1_u

        x1_zu = self.bn(x1_zu)
        x1_zu = self.relu(x1_zu)
        x1 = self.inc_zu(x1_zu)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        out = 10 * torch.tanh((zeta + x) / 10)

        return out





