import torch
from escnn import gspaces
from escnn import nn
import numpy as np

r2_act = gspaces.rot2dOnR2(N=4)
# r2_act = gspaces.flipRot2dOnR2(N=4)
# r2_act = gspaces.flip2dOnR2(axis=np.pi/4)
print(r2_act.irreps)

import matplotlib.pyplot as plt
import numpy as np
feat_type_in = nn.FieldType(r2_act, [r2_act.irrep(1)])         # rot
# feat_type_in = nn.FieldType(r2_act, [r2_act.irrep(1,1)])   # flipRot
# feat_type_in = nn.FieldType(r2_act, [r2_act.irrep(0)])   # flip

feat_type_hid = nn.FieldType(r2_act, 8 * [r2_act.regular_repr])

feat_type_out = nn.FieldType(r2_act, [r2_act.irrep(1)])       # rot
# feat_type_out = nn.FieldType(r2_act, [r2_act.irrep(1,1)])  # flipRot
# feat_type_out = nn.FieldType(r2_act, [r2_act.irrep(0)])  # flip

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
x1 = torch.randn(1, 1, S, S)
x2 = torch.randn(1, 1, S, S)

x = torch.cat([x1, x2], dim=1)

x = feat_type_in(x)

print(x.size())

fig, axs = plt.subplots(1, r2_act.fibergroup.order(), sharex=True, sharey=True, figsize=(16, 4))
X, Y = np.meshgrid(range(S - 6), range(S - 7, -1, -1))

# for each group element
for i, g in enumerate(r2_act.testing_elements):
    # transform the input
    x_transformed = x.transform(g)

    y = model(x_transformed)
    print(y.size())
    y = y.tensor.detach().numpy().squeeze()

    # plot the output vector field
    axs[i].quiver(X, Y, y[0, ...], y[1, ...], units='xy')
    axs[i].set_title(g.to('int') * 90)

plt.show()