# Code for generating indices used in G-convolutions for various groups G.
# The indices created by these functions are used to rotate and flip filters on the plane or on a group.
# These indices depend only on the filter size, so they are created only once at the beginning of training.

import torch

###########################################
# generate indices for 1D mirror reflection
# special ksize = 6 for u, ksize = 7 for z
###########################################
def make_m2_z1_indices_z(ksize):
    generate_inds = torch.tensor([[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
                                  [[0, 6], [0, 5], [0, 4], [0, 3], [0, 2], [0, 1], [0, 0]]])
    generate_inds = generate_inds[:, None, None, :, :]
    generate_inds = generate_inds.numpy()
    return generate_inds.astype('int32')

def make_m2_z1_indices_u(ksize):
    generate_inds = torch.tensor([[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
                                  [[0, 5], [0, 4], [0, 3], [0, 2], [0, 1], [0, 0]]])
    generate_inds = generate_inds[:, None, None, :, :]
    generate_inds = generate_inds.numpy()
    return generate_inds.astype('int32')


def make_m2_m2_indices(ksize):
    generate_inds = torch.tensor([[[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
                                   [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]],
                                  [[[1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1], [1, 0]],
                                   [[0, 6], [0, 5], [0, 4], [0, 3], [0, 2], [0, 1], [0, 0]]]])

    generate_inds = generate_inds[:, :, None, :, :]
    generate_inds = generate_inds.numpy()
    return generate_inds.astype('int32')


