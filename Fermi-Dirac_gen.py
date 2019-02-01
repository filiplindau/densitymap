"""
Created on 28 Jan 2019

@author: Filip Lindau
"""

import numpy as np
from pyDOE import lhs
import ghalton
import time
from scipy.interpolate import interp1d
import scipy.optimize as so
import logging
from matplotlib.pyplot import imread

logger = logging.getLogger()
while logger.handlers:
    logger.removeHandler(logger.handlers[0])
f = logging.Formatter("%(asctime)s - %(module)s. %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


class FD(object):
    def __init__(self):
        self.f = None
        self.f3d = None
        self.F1 = None
        self.F1_interp = None
        self.E_min = None
        self.E_max = None
        self.th_max = None

        self.E_photon = None
        self.E_work = None
        self.E_F = None

    def gen_FD_CDF(self, E_photon, E_work, E_F):
        logger.info("Generating FD CDF with E_ph={0} eV, E_w={1} eV, E_F={2} eV".format(E_photon, E_work, E_F))
        qe = 1.602e-19
        me = 9.11e-31

        self.E_F = E_F
        self.E_photon = E_photon
        self.E_work = E_work

        E = np.linspace(E_F + E_work, E_F + E_photon, 1000)
        th_max = np.arccos(np.sqrt(E.min()/E.max()))
        self.E_min = E.min()
        self.E_max = E.max()
        self.th_max = th_max
        th = np.linspace(0, th_max, 500)
        thm, Em = np.meshgrid(th, E)
        th_max_m = np.arccos(np.sqrt(E.min() / Em))
        self.f = np.zeros_like(Em)
        self.f[thm < th_max_m] = 1
        F = self.f.cumsum(0).cumsum(1)
        F /= F.max()

        self.F1 = self.f.sum(1).cumsum()
        self.F1 /= self.F1.max()

        x = np.arange(self.F1.shape[0])
        self.F1_interp = interp1d(x, self.F1)

    def gen_fd_3d(self, E_photon, E_work, E_F):
        logger.info("Generating FD 3D DF with E_ph={0} eV, E_w={1} eV, E_F={2} eV".format(E_photon, E_work, E_F))
        qe = 1.602e-19
        me = 9.11e-31

        self.E_F = E_F
        self.E_photon = E_photon
        self.E_work = E_work

        E = np.linspace(E_F + E_work, E_F + E_photon, 1000)
        th_max = np.arccos(np.sqrt(E.min()/E.max()))
        self.E_min = E.min()
        self.E_max = E.max()
        self.th_max = th_max
        th = np.linspace(0, th_max, 500)
        phi = np.linspace(0, np.pi*2, 250)
        phim, thm, Em = np.meshgrid(phi, th, E)
        th_max_m = np.arccos(np.sqrt(E.min() / Em))
        self.f3d = np.zeros_like(Em)
        self.f3d[thm < th_max_m] = 1

    def sample_FD(self, random_vec):
        qe = 1.602e-19
        me = 9.11e-31
        c = 299792458.0

        F = self.f.sum(1).cumsum()
        F /= F.max()

        x = np.interp(random_vec[:, 0], F, np.arange(F.shape[0]))
        E = self.E_min + x * (self.E_max - self.E_min) / self.f.shape[0]

        th = random_vec[:, 1] * np.arccos(np.sqrt(self.E_min / E))
        p_par = np.sqrt(2 * me * E * qe) * np.sin(th) * c / qe
        p_z = np.sqrt(2 * me * (E * np.cos(th)**2 - self.E_min) * qe) * c / qe

        phi = random_vec[:, 2] * 2 * np.pi
        p_x = p_par * np.cos(phi)
        p_y = p_par * np.sin(phi)

        # p = np.hstack((p_par.reshape((-1, 1)), p_z.reshape((-1, 1))))
        p = np.hstack((p_x.reshape((-1, 1)), p_y.reshape((-1, 1)), p_z.reshape((-1, 1))))

        # self.E = E
        # self.th = th

        return p

    def sample_fd_cos(self, random_vec):
        qe = 1.602e-19
        me = 9.11e-31
        c = 299792458.0

        E_F = self.E_F
        E_photon = self.E_photon
        E_work = self.E_work

        E = np.linspace(E_F + E_work, E_F + E_photon, 1000)
        costh_min = np.sqrt(E.min() / E.max())
        costh = np.linspace(costh_min, 1, 500)
        costhm, Em = np.meshgrid(costh, E)
        costh_min_m = np.sqrt(E.min() / Em)
        f = np.zeros_like(Em)
        f[costhm > costh_min_m] = 1
        self.f_cos = f

        F = f.sum(1).cumsum()
        F /= F.max()

        x = np.interp(random_vec[:, 0], F, np.arange(F.shape[0]))
        E = E.min() + x * (E.max() - E.min()) / f.shape[0]

        costh = np.sqrt(E.min() / E) + random_vec[:, 1] * (1 - np.sqrt(E.min() / E))
        p_par = np.sqrt(2 * me * E * qe) * np.sqrt(1 - costh**2) * c / qe
        p_z = np.sqrt(2 * me * (E * costh ** 2 - E.min()) * qe) * c / qe

        phi = random_vec[:, 2] * 2 * np.pi
        p_x = p_par * np.cos(phi)
        p_y = p_par * np.sin(phi)

        # p = np.hstack((p_par.reshape((-1, 1)), p_z.reshape((-1, 1))))
        p = np.hstack((p_x.reshape((-1, 1)), p_y.reshape((-1, 1)), p_z.reshape((-1, 1))))

        return p

    def sample_fd_sin(self, random_vec):
        logger.info("Sampling FD distribution for {0} particles using sin theta.".format(random_vec.shape[0]))
        qe = 1.602e-19
        me = 9.11e-31
        c = 299792458.0

        E_F = self.E_F
        E_photon = self.E_photon
        E_work = self.E_work

        E = np.linspace(E_F + E_work, E_F + E_photon, 1000)
        th_max = np.arccos(np.sqrt(E.min() / E.max()))
        th = np.linspace(0, th_max, 500)
        thm, Em = np.meshgrid(th, E)
        th_max_m = np.arccos(np.sqrt(E.min() / Em))
        f = np.sin(thm)
        f[thm > th_max_m] = 0
        self.f_sin = f

        F = f.sum(1).cumsum()
        F /= F.max()

        x = np.interp(random_vec[:, 0], F, np.arange(F.shape[0]))
        Es = E.min() + x * (E.max() - E.min()) / f.shape[0]

        ths = random_vec[:, 1] * np.arccos(np.sqrt(E.min() / Es))
        p_par = np.sqrt(2 * me * Es * qe) * np.sin(ths) * c / qe
        p_z = np.sqrt(2 * me * (Es * np.cos(ths) ** 2 - E.min()) * qe) * c / qe

        phi = random_vec[:, 2] * 2 * np.pi
        p_x = p_par * np.cos(phi)
        p_y = p_par * np.sin(phi)

        # p = np.hstack((p_par.reshape((-1, 1)), p_z.reshape((-1, 1))))
        p = np.hstack((p_x.reshape((-1, 1)), p_y.reshape((-1, 1)), p_z.reshape((-1, 1))))

        return p

    def sample_FD_CDF(self, random_vec):
        logger.info("Sampling FD distribution for {0} particles.".format(random_vec.shape[0]))
        qe = 1.602e-19
        me = 9.11e-31
        c = 299792458.0

        x_data = random_vec
        # Loop through particles
        xi = []  # Index list

        for k in range(x_data.shape[0]):
            xi_high = (self.F1 > x_data[k, 0]).searchsorted(True)
            dfdx = self.F1[xi_high] - self.F1[xi_high - 1]  # dx = 1
            df = self.F1[xi_high] - x_data[k, 0]
            dx = df / dfdx
            xi_1 = xi_high - dx

            F2 = self.f[xi_high, :].cumsum()
            F2 /= F2.max()
            xi_high = (F2 > x_data[k, 1]).searchsorted(True)
            dfdy = F2[xi_high] - F2[xi_high - 1]  # dx = 1
            df = F2[xi_high] - x_data[k, 1]
            dy = df / dfdy
            xi_2 = xi_high - dy

            E = self.E_min + xi_1 * (self.E_max - self.E_min) / self.F1.shape[0]
            th = xi_2 * self.th_max / self.f.shape[1]

            p_par = np.sqrt(2 * me * E * qe) * np.sin(th) * c / qe
            th_out = np.arcsin(np.sqrt(E/(E-self.E_min + 1e-9)) * np.sin(th))
            p_z = p_par / np.tan(th_out)

            # xi.append(np.array([xi_1, xi_2]))
            # xi.append(np.array([E, th]))
            xi.append(np.array([p_par, p_z]))

        p = np.array(xi)
        return p

    def sample_2d_pdf(self, pdf, points, xlim, ylim):

        # Create CDF in axis 0 direction by summing in axis 1, then cumsum:
        F = pdf.sum(1).cumsum()
        F /= F.max()

        x = np.interp(points[:, 0], F, np.arange(F.shape[0]))
        xi = np.around(x).astype(np.int)        # For indexing

        F2 = pdf.cumsum(axis=1)
        F2 /= F2.max(axis=1).reshape((-1, 1)).repeat(F2.shape[1], axis=1)

        yi = np.greater(F2[xi, :], points[:, 1].reshape((-1, 1))).argmax(axis=1)
        y = yi-(F2[xi, yi]-points[:, 1])/(F2[xi, yi]-F2[xi, yi-1])          # Interpolation

        px = xlim[0] + x * (xlim[1] - xlim[0]) / pdf.shape[0]
        py = ylim[0] + y * (ylim[1] - ylim[0]) / pdf.shape[1]
        p = np.hstack((px.reshape((-1, 1)), py.reshape((-1, 1))))

        return p

    def sample_3d_pdf(self, pdf, points, xlim, ylim, zlim):
        logger.info("Sampling FD distribution for {0} particles.".format(random_vec.shape[0]))
        # Create CDF in axis 0 direction by summing in axis 1, then cumsum:
        F = pdf.sum(2).sum(1).cumsum()
        F /= F.max()

        x = np.interp(points[:, 0], F, np.arange(F.shape[0]))
        xi = np.around(x).astype(np.int)        # For indexing

        F2 = pdf.sum(2).cumsum(axis=1)
        F2 /= F2.max(axis=1).reshape((-1, 1)).repeat(F2.shape[1], axis=1)

        yi = np.greater(F2[xi, :], points[:, 1].reshape((-1, 1))).argmax(axis=1)
        y = yi-(F2[xi, yi]-points[:, 1])/(F2[xi, yi]-F2[xi, yi-1])          # Interpolation

        F3 = pdf.cumsum(axis=2)
        F3 /= F3.max(axis=2).reshape((F3.shape[0], F3.shape[1], 1)).repeat(F3.shape[2], axis=2)

        zi = np.greater(F3[xi, yi, :], points[:, 2].reshape((-1, 1))).argmax(axis=1)
        z = zi-(F3[xi, yi, zi]-points[:, 2])/(F3[xi, yi, zi]-F3[xi, yi, zi-1])          # Interpolation

        px = xlim[0] + x * (xlim[1] - xlim[0]) / pdf.shape[0]
        py = ylim[0] + y * (ylim[1] - ylim[0]) / pdf.shape[1]
        pz = zlim[0] + z * (zlim[1] - zlim[0]) / pdf.shape[2]
        p = np.hstack((px.reshape((-1, 1)), py.reshape((-1, 1)), pz.reshape((-1, 1))))

        return p


n_points = 20e3

fd = FD()
fd.gen_FD_CDF(4.71, 4.46, 7.0)

halton_sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:3])
random_vec = np.array(halton_sequencer.get(int(n_points)))

# p = fd.sample_FD_CDF(random_vec)

# p = fd.sample_2d_pdf(fd.f, random_vec, [fd.E_min, fd.E_max], [0, fd.th_max])

pt = fd.sample_FD(random_vec)

p = fd.sample_fd_sin(random_vec)
pc = fd.sample_fd_cos(random_vec)
ps = p
p_par = np.sqrt(p[:, 0]**2 + p[:, 1]**2)


# 3D:
# fd.gen_fd_3d(4.71, 4.46, 7.0)
# p = fd.sample_3d_pdf(fd.f3d, random_vec, [fd.E_min, fd.E_max], [0, fd.th_max], [0, 2*np.pi])

