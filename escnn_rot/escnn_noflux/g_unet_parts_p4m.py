"""
Parts of the U-Net model by Ronneberger et Al.
taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""

import torch
from escnn import nn
from pooling import *
from escnn.nn import *
from escnn.gspaces import *

# import torch.nn.functional as F

class DoubleConv1(torch.nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super(DoubleConv1, self).__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.conv01 = nn.R2Conv(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode="circular")
		self.bn01 = nn.InnerBatchNorm(mid_channels)
		self.conv02 = nn.R2Conv(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")
		self.bn02 = nn.InnerBatchNorm(out_channels)
		self.relu = nn.ReLU(out_channels)

	def forward(self, x):
		out = self.relu(self.bn01(self.conv01(x)))
		out = self.relu(self.bn02(self.conv02(out)))
		return out


class DoubleConv(torch.nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.conv1 = nn.R2Conv(in_channels, mid_channels, kernel_size=3, padding=1,  padding_mode="circular")
		self.bn1 = nn.InnerBatchNorm(mid_channels)
		self.conv2 = nn.R2Conv(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")
		self.bn2 = nn.InnerBatchNorm(out_channels)
		self.relu = nn.ReLU(out_channels)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.relu(self.bn2(self.conv2(out)))
		return out

class Down(torch.nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.normMaxPool = nn.NormMaxPool(in_channels, kernel_size=(2, 2))
		self.conv1 = DoubleConv(in_channels, out_channels)

	def forward(self, x):
		out = self.normMaxPool(x)
		out = self.conv1(out)
		return out

class Up1(torch.nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, mid_channels, out_channels, trilinear=True):
		super().__init__()
		self.up = R2Upsampling(mid_channels, scale_factor=2)
		self.conv = DoubleConv(in_channels, out_channels)
		self.in_channels = in_channels
	def forward(self, x1, x2):  # (x5, x4)
		x1 = self.up(x1)
		x = torch.cat([x2.tensor, x1.tensor], dim=1)
		x = GeometricTensor(x, self.in_channels)
		return self.conv(x)

#################################
class Up2(torch.nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels, trilinear=True):
		super().__init__()
		self.up = R2Upsampling(mid_channels, scale_factor=2)
		self.conv = DoubleConv(in_channels, out_channels)
		self.in_channels = in_channels
	def forward(self, x1, x2):
		x1 = self.up(x1)
		x = torch.cat([x2.tensor, x1.tensor], dim=1)
		x = GeometricTensor(x, self.in_channels)
		return self.conv(x)

#################################
class Up3(torch.nn.Module):
	def __init__(self, in_channels, mid_channels,out_channels, trilinear=True):
		super().__init__()
		self.up = R2Upsampling(mid_channels, scale_factor=2)
		self.conv = DoubleConv(in_channels, out_channels)
		self.in_channels = in_channels
	def forward(self, x1, x2):
		x1 = self.up(x1)
		x = torch.cat([x2.tensor, x1.tensor], dim=1)
		x = GeometricTensor(x, self.in_channels)
		return self.conv(x)

#################################
class Up4(torch.nn.Module):
	def __init__(self, in_channels,mid_channels, out_channels, trilinear=True):
		super().__init__()
		self.up = R2Upsampling(mid_channels, scale_factor=2)
		self.conv = DoubleConv(in_channels, out_channels)
		self.in_channels = in_channels

	def forward(self, x1, x2):
		x1 = self.up(x1)
		x = torch.cat([x2.tensor, x1.tensor], dim=1)
		x = GeometricTensor(x, self.in_channels)
		return self.conv(x)


class OutConv(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.R2Conv(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		x1 = self.conv(x)
		x1 = x1.tensor

		hu = x1[:, 0, :, :]
		hv = x1[:, 1, :, :]

		return hu, hv