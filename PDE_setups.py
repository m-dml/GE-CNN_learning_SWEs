import torch
import numpy as np

class CNN_dataset:
    def __init__(self, nx, nx_u, dx, dt, cd, g, wimp, batch_size=100, dataset_size=1000, average_sequence_length=5000):
        self.nx = nx
        self.nx_u = nx_u
        self.dx = dx
        self.dt = dt
        self.cd = cd
        self.g = g
        self.wimp = wimp
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.average_sequence_length = average_sequence_length
        self.u = torch.zeros(dataset_size, 1, nx_u, dtype=torch.float32)
        self.zeta = torch.zeros(dataset_size, 1, nx, dtype=torch.float32)
        self.h = torch.zeros(dataset_size, 1, nx, dtype=torch.float32)
        self.d = torch.ones(dataset_size, 1, nx, dtype=torch.float32) * 100  # undisturbed water depth [m]
        self.d[:, :, 0] = -10
        self.d[:, :, -1] = -10

        for i in range(dataset_size):
            self.reset_env(i)

        size_boundary = 1
        self.boundary_mask_z = torch.ones(dataset_size, 1, nx, dtype=torch.float32)
        self.boundary_mask_z[:, :, size_boundary:-size_boundary] = 0

        self.zeta_mask = torch.ones(dataset_size, 1, nx, dtype=torch.float32)
        self.zeta_mask[:, :, 0:size_boundary] = 0
        self.zeta_mask[:, :, -size_boundary:] = 0

        self.boundary_mask_u = torch.ones(dataset_size, 1, nx_u, dtype=torch.float32)
        self.boundary_mask_u[:, :, size_boundary:-size_boundary] = 0

        self.u_mask = torch.ones(dataset_size, 1, nx_u, dtype=torch.float32)
        self.u_mask[:, :, 0:size_boundary] = 0
        self.u_mask[:, :, -size_boundary:] = 0

        self.t = 0
        self.i = 0

        self.us = torch.zeros(dataset_size, 1, nx_u, dtype=torch.float32)
        self.uint = torch.zeros(dataset_size, 1, nx_u, dtype=torch.float32)
        self.uints = torch.zeros(dataset_size, 1, nx_u, dtype=torch.float32)
        self.ce = torch.zeros(dataset_size, 1, nx, dtype=torch.float32)
        self.cw = torch.zeros(dataset_size, 1, nx, dtype=torch.float32)
        self.div = torch.zeros(dataset_size, 1, nx, dtype=torch.float32)

    def reset_env(self, index):
        """
        """
        # Random position and random value for the initial perturbation and "cold starts"
        x = torch.arange(10, 190)
        mean = np.random.randint(10, 190, 1)
        std = np.random.randint(1, 10, 1)
        std = torch.from_numpy(std)
        z = 1 / (std * (2 * torch.pi) ** 0.5) * torch.exp(-0.5 * ((x - mean) / std) ** 2)
        self.zeta[index, :, :] = 0
        self.zeta[index, 0, 10:190] = z

        self.h[index, 0, :] = self.zeta[index, 0, :] + self.d[index, 0, :]
        self.u[index, 0, :] = torch.zeros(self.nx_u, dtype=torch.float32)

    def ask(self):
        """
        ask for a batch of boundary and initial conditions
        :return: u, zeta, h
        """
        self.indices = np.random.choice(self.dataset_size, self.batch_size)
        return self.zeta[self.indices, :, :], self.u[self.indices, :, :], self.h[self.indices, :, :], \
               self.boundary_mask_z[self.indices, :, :], self.zeta_mask[self.indices, :, :], \
               self.boundary_mask_u[self.indices, :, :], self.u_mask[self.indices, :, :], \
               self.d[self.indices, :, :], self.us[self.indices, :, :]

    def tell(self, zeta, u, h):
        """
        return the updated fluid state (a and p) to the dataset
        """
        self.u[self.indices, :, :] = u.detach()
        self.zeta[self.indices, :, :] = zeta.detach()
        self.h[self.indices, :, :] = h.detach()

        self.t += 1
        if self.t % (self.average_sequence_length / self.batch_size) == 0:
            self.reset_env(int(self.i))
            self.i = (self.i + 1) % self.dataset_size


    # ask and tell the data for the calculation of Loss
    def generate_masked_tensor(self, input, mask, fill=0):
        masked_tensor = torch.zeros(input.size()) + fill
        masked_tensor[mask] = input[mask]
        return masked_tensor

    def askMb(self):
        ii = torch.arange(0, self.nx - 1, dtype=torch.int)
        ip = torch.arange(1, self.nx, dtype=torch.int)

        hh = self.h[self.indices, :, :]
        hm = 0.5 * (hh[:, :, ii.numpy()] + hh[:, :, ip.numpy()])

        zeta_z = self.zeta[self.indices, :, :]
        dz_dx = (zeta_z[:, :, ip.numpy()] - zeta_z[:, :, ii.numpy()]) / self.dx

        self.us[self.indices, :, :] = self.u[self.indices, :, :] - self.dt * self.cd / hm * self.u[self.indices, :, :] *\
                                      torch.abs(self.u[self.indices, :, :]) - (1 - self.wimp) * self.dt * self.g * dz_dx

        self.uint[self.indices, :, :] = hm * self.u[self.indices, :, :]
        self.uints[self.indices, :, :] = hm * self.us[self.indices, :, :]


        ## for calculate mask
        dd = self.d[self.indices, :, :]
        mask_u = torch.logical_and(dd[:, :, ii.numpy()] > 0.1, dd[:, :, ip.numpy()] > 0.1)

        self.us[self.indices, :, :] = self.generate_masked_tensor(self.us[self.indices, :, :], mask_u)
        self.uint[self.indices, :, :] = self.generate_masked_tensor(self.uint[self.indices, :, :], mask_u)
        self.uints[self.indices, :, :] = self.generate_masked_tensor(self.uints[self.indices, :, :], mask_u)

        ######################
        # calculate div
        ######################
        i_u = torch.arange(0, self.nx_u, dtype=torch.int)  # an index for u = 0:198
        i_u_r = torch.tensor([self.nx_u - 1], dtype=torch.int)  # an right index 198 for u
        ii_u = torch.cat((i_u, i_u_r))  # an index for right u =0:198,198

        im2_u = torch.tensor([0], dtype=torch.int)  # an left index 0 for u
        im_u = torch.cat((im2_u, i_u))  # an index for left u =0, 0:198

        ut = self.uint[self.indices, :, :]
        uts = self.uints[self.indices, :, :]

        du_dx = (ut[:, :, ii_u.numpy()] - ut[:, :, im_u.numpy()]) / self.dx
        dus_dx = (uts[:, :, ii_u.numpy()] - uts[:, :, im_u.numpy()]) / self.dx

        self.div[self.indices, :, :] = -self.dt * (1-self.wimp) * du_dx - self.dt * self.wimp * dus_dx

        # use a mask
        ii_z = torch.arange(0, self.nx, dtype=torch.int)  # an total index for z = 0:199
        mask1 = dd[:, :, ii_z.numpy()].ge(0.1)
        self.div[self.indices, :, :] = self.generate_masked_tensor(self.div[self.indices, :, :], mask1)


        #########################
        #      ce and cw
        #########################
        ii_ew = torch.arange(0, self.nx, dtype=torch.int)

        ip1_ew = torch.arange(1, self.nx, dtype=torch.int)
        ip2_ew = torch.tensor([self.nx - 1], dtype=torch.int)
        ip_ew = torch.cat((ip1_ew, ip2_ew))

        im1_ew = torch.arange(0, self.nx - 1, dtype=torch.int)
        im2_ew = torch.tensor([0], dtype=torch.int)
        im_ew = torch.cat((im2_ew, im1_ew))


        hip_dx2 = (hh[:, :, ii_ew.numpy()] + hh[:, :, ip_ew.numpy()]) / self.dx**2
        self.ce[self.indices, :, :] = self.dt ** 2 * self.wimp ** 2 * self.g * 0.5 * hip_dx2

        him_dx2 = (hh[:, :, ii_ew.numpy()] + hh[:, :, im_ew.numpy()]) / self.dx ** 2
        self.cw[self.indices, :, :] = self.dt ** 2 * self.wimp ** 2 * self.g * 0.5 * him_dx2

        mask_ce = torch.logical_and(dd[:, :, ii_ew.numpy()] > 0.1, dd[:, :, ip_ew.numpy()] > 0.1)
        mask_cw = torch.logical_and(dd[:, :, ii_ew.numpy()] > 0.1, dd[:, :, im_ew.numpy()] > 0.1)

        self.ce[self.indices, :, :] = self.generate_masked_tensor(self.ce[self.indices, :, :], mask_ce)
        self.cw[self.indices, :, :] = self.generate_masked_tensor(self.cw[self.indices, :, :], mask_cw)

        M = torch.ones(self.nx)
        M = M.repeat(self.batch_size, 1)
        M = M[:, None, :]

        up1 = - self.ce[self.indices, :, :] / (1 + self.ce[self.indices, :, :] + self.cw[self.indices, :, :])
        up1 = up1[:, :, 0:-1]

        down1 = - self.cw[self.indices, :, :] / (1 + self.ce[self.indices, :, :] + self.cw[self.indices, :, :])
        down1 = down1[:, :, 1:]

        b = (self.zeta[self.indices, :, :] + self.div[self.indices, :, :]) / (1 + self.ce[self.indices, :, :] + self.cw[self.indices, :, :])

        b[:, :, 0] = 0.0
        b[:, :, -1] = 0.0

        return M.cuda(), up1.cuda(), down1.cuda(), b.cuda()

    def askuh(self, zetan):
        zetan = zetan.detach()

        ii = torch.arange(0, self.nx - 1, dtype=torch.int)
        ip = torch.arange(1, self.nx, dtype=torch.int)

        dzn_dx = (zetan[:, :, ip.numpy()] - zetan[:, :, ii.numpy()]) / self.dx

        dd = self.d[self.indices, :, :]

        mask_u = torch.logical_and(dd[:, :, ii.numpy()] > 0.1, dd[:, :, ip.numpy()] > 0.1)

        self.u[self.indices, :, :] = self.us[self.indices, :, :] - self.wimp * self.dt * self.g * dzn_dx
        self.u[self.indices, :, :] = self.generate_masked_tensor(self.u[self.indices, :, :], mask_u)


        ii_z = torch.arange(0, self.nx, dtype=torch.int)
        mask1 = dd[:, :, ii_z.numpy()].ge(0.1)
        self.h[self.indices, :, :] = self.d[self.indices, :, :] + zetan
        self.h[self.indices, :, :] = self.generate_masked_tensor(self.h[self.indices, :, :], mask1)

        return self.u[self.indices, :, :].cuda(), self.h[self.indices, :, :].cuda()





