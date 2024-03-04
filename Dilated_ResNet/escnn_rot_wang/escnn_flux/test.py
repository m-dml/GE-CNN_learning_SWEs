import torch
from escnn import gspaces
from escnn import nn


r2_act = gspaces.rot2dOnR2(N=4)

# feat_type_in = nn.FieldType(r2_act, 2*[r2_act.trivial_repr])
# # feat_type_out = nn.FieldType(r2_act, [r2_act.regular_repr])
# feat_type_out = nn.FieldType(r2_act, 3*[r2_act.regular_repr])
# conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=3)

import matplotlib.pyplot as plt
import numpy as np

feat_type_in = nn.FieldType(r2_act, [r2_act.trivial_repr])
feat_type_hid = nn.FieldType(r2_act, 8 * [r2_act.regular_repr])
feat_type_out = nn.FieldType(r2_act, [r2_act.irrep(1)])




model = nn.SequentialModule(
    nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=3),
    nn.InnerBatchNorm(feat_type_hid),
    nn.ReLU(feat_type_hid),
    nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=3),
    nn.InnerBatchNorm(feat_type_hid),
    nn.ReLU(feat_type_hid),
    nn.R2Conv(feat_type_hid, feat_type_out, kernel_size=3),
).eval()




S = 11
x = torch.randn(1, 1, S, S)
x = feat_type_in(x)

fig, axs = plt.subplots(1, r2_act.fibergroup.order(), sharex=True, sharey=True, figsize=(16, 4))

X, Y = np.meshgrid(range(S - 6), range(S - 7, -1, -1))

# for each group element
for i, g in enumerate(r2_act.testing_elements):
    # transform the input
    x_transformed = x.transform(g)

    y = model(x_transformed)
    y = y.tensor.detach().numpy().squeeze()

    # plot the output vector field
    axs[i].quiver(X, Y, y[0, ...], y[1, ...], units='xy')
    axs[i].set_title(g.to('int') * 90)

plt.show()