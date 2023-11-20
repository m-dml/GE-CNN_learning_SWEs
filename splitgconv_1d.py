"""
Parts of changed group P4 by Cohen et Al.
changed from: https://github.com/jornpeters/GrouPy/tree/pytorch\_p4\_p4m\_gconv/groupy/gconv
"""
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from make_gconv_indices_1d import *
import numpy as np

make_indices_functions = {(1, 2, 0): make_m2_z1_indices_z,
                          (1, 2, 1): make_m2_z1_indices_u,
                          (2, 2, 2): make_m2_m2_indices}

def trans_filter(w, inds):
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist()]

    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    return w_transformed.contiguous()


class SplitGConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_stabilizer_size=1, output_stabilizer_size=2, different_channel=0,
                 velocity_inputs=None):
        super(SplitGConv1D, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size, different_channel) in make_indices_functions.keys()
        self.ksize = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(different_channel)

        self.inds = self.make_transformation_indices(different_channel)

        self.velocity_inputs = velocity_inputs
        self.different_channel = different_channel


        if velocity_inputs is not None:
            assert input_stabilizer_size == 1,           "velocity inputs are supported only for input layers (Z1->M2)"
            assert len(velocity_inputs) == in_channels,  "invalid input, velocity channels must be a bool array with in_channels elements"
            self.velocity_inputs = torch.tensor(self.velocity_inputs, dtype=torch.bool, requires_grad=False)
            filter_sign_shape = (1, 2, self.in_channels, 1, 1, self.ksize)
            self.filter_sign = torch.ones(filter_sign_shape, dtype=self.weight.dtype, requires_grad=False)
            self.filter_sign[:,  # all output channels (broadcasted)
                             1,  # flipped filter only
                             self.velocity_inputs,  # velocity input channels only
                             :,  # no flipping of inputs is possible for input layers Z1->M2
                             :,  # rows (only 1 row for 1D)
                             :   # columns (filter elements)
                             ] = -1.0


    def reset_parameters(self, different_channel):
        if different_channel == 0:
            n = self.in_channels
            k = 6
            n *= k
            stdv = 1. / math.sqrt(n)
        elif different_channel == 1:
            n = self.in_channels
            k = 7
            n *= k
            stdv = 1. / math.sqrt(n)
        else:
            n = self.in_channels
            k = 8
            n *= k
            stdv = 1. / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self, different_channel):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size, different_channel)](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds)

        if self.velocity_inputs is not None:
            self.filter_sign = self.filter_sign.cuda()
            tw = tw * self.filter_sign


        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize)
        tw = tw.view(tw_shape)

        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-1])

        y = F.conv1d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)

        batch_size, _, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            y = y + bias
        return y

class M2ConvZ1_z(SplitGConv1D):
    def __init__(self, *args, **kwargs):
        super(M2ConvZ1_z, self).__init__(input_stabilizer_size=1, output_stabilizer_size=2, different_channel=0,
                                       velocity_inputs=[False, False, False, False, False,
                                                        False, False, False], *args, **kwargs)

class M2ConvZ1_u(SplitGConv1D):
    def __init__(self, *args, **kwargs):
        super(M2ConvZ1_u, self).__init__(input_stabilizer_size=1, output_stabilizer_size=2, different_channel=1,
                                       velocity_inputs=[True, True, True, True, True,
                                                        ], *args, **kwargs)

class M2ConvM2(SplitGConv1D):
    def __init__(self, *args, **kwargs):
        super(M2ConvM2, self).__init__(input_stabilizer_size=2, output_stabilizer_size=2, different_channel=2,
                                       velocity_inputs=None, *args, **kwargs)
