"""
Parts of the U-Net model by Ronneberger et Al.
taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from splitgconv_1d import M2ConvZ1_z, M2ConvZ1_u, M2ConvM2
from pooling_1d import plane_group_spatial_max_pooling
import pytorch_lightning as pl

class Conv_z(pl.LightningModule):
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super(Conv_z, self).__init__()
		self.conv01 = M2ConvZ1_z(in_channels, out_channels, kernel_size=7, padding=3)
	def forward(self, x):
		out = self.conv01(x)
		return out

class Conv_u(pl.LightningModule):
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super(Conv_u, self).__init__()
		self.conv01 = M2ConvZ1_u(in_channels, out_channels, kernel_size=6, padding=3)
	def forward(self, x):
		out = self.conv01(x)
		return out

class Conv_zu(pl.LightningModule):
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super(Conv_zu, self).__init__()
		self.conv02 = M2ConvM2(in_channels, out_channels, kernel_size=7, padding=3)
		self.bn02 = nn.BatchNorm2d(out_channels)
	def forward(self, x):
		out = F.relu(self.bn02(self.conv02(x)))
		return out

class DoubleConv(pl.LightningModule):
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.conv1 = M2ConvM2(in_channels, mid_channels, kernel_size=7, padding=3)
		self.bn1 = nn.BatchNorm2d(mid_channels)
		self.conv2 = M2ConvM2(mid_channels, out_channels, kernel_size=7, padding=3)
		self.bn2 = nn.BatchNorm2d(out_channels)
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		return out

class Down(pl.LightningModule):
	"""Downscaling with maxpool then double conv"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv1 = DoubleConv(in_channels, out_channels)

	def forward(self, x):
		out = plane_group_spatial_max_pooling(x, 2, 2)
		out = self.conv1(out)
		return out

class Up1(pl.LightningModule):
	"""Upscaling then double conv"""
	def __init__(self, in_channels, out_channels, trilinear=True):
		super().__init__()

		# if trilinear, use the normal convolutions to reduce the number of channels
		if trilinear:
			# self.up = nn.Upsample(scale_factor=2, mode='nearest')
			self.up = nn.Upsample(size =(2, 12), mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

#################################
class Up2(pl.LightningModule):
	"""Upscaling then double conv"""
	def __init__(self, in_channels, out_channels, trilinear=True):
		super().__init__()
		if trilinear:
			self.up = nn.Upsample(size =(2, 25), mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)
	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class Up3(pl.LightningModule):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, trilinear=True):
		super().__init__()

		if trilinear:
			self.up = nn.Upsample(size =(2, 50), mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

#################################
class Up4(pl.LightningModule):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, trilinear=True):
		super().__init__()
		if trilinear:
			self.up = nn.Upsample(size =(2, 100), mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(pl.LightningModule):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = M2ConvM2(in_channels, out_channels, kernel_size=7, padding=3)
	def forward(self, x):
		x1 = self.conv(x)
		x1 = torch.mean(x1, 2)
		return x1