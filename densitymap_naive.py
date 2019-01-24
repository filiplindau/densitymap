"""
Created on 06 Jul 2016

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
logger.setLevel(logging.DEBUG)


class DensityMapNaive(object):
    def __init__(self, image=None):
        self.image = None
        self.image_pixel_resolution = 30e-6
        self.size = None
        self.F1 = None
        self.F1_interp = None
        self.image_cdf = None
        self.gen = 'halton'
        if image is not None:
            self._set_image(image)

        self.time_structure = None
        self.time_resolution = 10e-15
        self.t = None   # Variables for charge distribution in time
        self.t_int = None
        self.t_cdf = None

        self.momentum_image_pixel_resolution = None
        self.momentum_image = None
        self.momentum_image_pixel_resolution = 30e-6
        self.momentum_size = None
        self.momentum_F1 = None
        self.momentum_F1_interp = None
        self.momentum_image_cdf = None

        self.image_fd = None
        self.F1_fd = None
        self.F1_fd_interp = None
        self.image_fd_cdf = None

        self.energy_structure = None
        self.energy_resolution = 10e-15
        self.energies = None   # Variables for charge distribution in time
        self.energy_int = None
        self.energy_cdf = None
        self.charge = 100e-12

        self.set_time_uniform(6e-12, 60e-15)
        self.set_transverse_gaussian(1.0e-3, 20e-5, 200)
        self.set_transverse_momentum_gaussian(1.0e-3, 20e-5, 200)
        self.set_energy_gaussian(0.5, 1.0, 0.001)
        self.set_charge(100e-12)

        self.halton_sequencer = None
        self.saved_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                                      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
                                      139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
                                      227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
                                      311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                                      401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487,
                                      491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593,
                                      599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659])

    def _set_image(self, image):
        """
        Set the internal density map image. The image will be normalized.

        :param image: 2D numpy array density map
        :return: nothing
        """
        pic = image.transpose()
        self.size = pic.shape

        max_p = pic.max()
        self.image = pic/max_p
        self._generate_x_cdf()

    def _set_momentum_image(self, image):
        """
        Set the internal density map momentum image. The image will be normalized.

        :param image: 2D numpy array density map
        :return: nothing
        """
        self.momentum_size = image.shape
        max_p = image.max()
        self.momentum_image = image / max_p
        self._generate_p_cdf()

    def set_transverse_gaussian(self, sigma, pixel_resolution, image_size=200):
        """ Generate a gaussian image for use as an electron distribution

        :param sigma: sigma of the gaussian
        :param pixel_resolution: Size of each pixel in the image (m)
        :param image_size: scalar number of pixels in each direction
        :return: nothing
        """
        logger.info("Setting transverse gaussian: sigma={0}, pixel resolution={1}, "
                    "image size={2}".format(sigma, pixel_resolution, image_size))
        x = np.arange(-image_size/2, image_size/2, dtype=np.int64) * pixel_resolution
        y = np.exp(-(x/np.double(sigma))**2)
        self.image_pixel_resolution = pixel_resolution
        image = np.outer(y, y)
        self._set_image(image)

    def set_transverse_tophat(self, radius, size):
        """
        Generate a top hat image for use as an electron distribution

        :param radius: radius if the top hat
        :param size: 2 element vector for the image dimensions
        :return:
        """
        logger.info("Setting transverse top hat: radius={0}, image size={1}".format(radius, size))
        image = np.zeros(size)
        x = np.arange(-size[0] / 2, size[0] / 2, dtype=np.int64)
        y = np.arange(-size[1] / 2, size[1] / 2, dtype=np.int64)
        xm, ym = np.meshgrid(x, y)
        good_ind = xm**2 + ym**2 < radius**2
        image[good_ind] = 1
        self._set_image(image)

    def set_transverse_image(self, image, pixel_resolution):
        logger.info("Setting transverse image: resolution={0}".format(pixel_resolution))
        self.image_pixel_resolution = pixel_resolution
        self._set_image(image)

    def set_time_uniform(self, time_duration, time_resolution):
        logger.info("Setting uniform time distribution: duration={0}, time resolution={1}".format(time_duration,
                                                                                                  time_resolution))
        t_rise = time_duration / 10
        nt_rise = np.int(t_rise / time_resolution)
        nt = np.int(1.0 * time_duration / time_resolution + 2 * nt_rise)
        self.time_resolution = time_resolution
        self.t = time_resolution * (np.arange(nt) - np.double(nt) / 2)
        self.time_structure = np.ones_like(self.t)
        self.time_structure[0:nt_rise] = np.linspace(0, 1, nt_rise)
        self.time_structure[nt-nt_rise:] = np.linspace(1, 0, nt_rise)
        self._generate_t_cdf()

    def set_time_gaussian(self, sigma, time_window, time_resolution=10e-15):
        logger.info("Setting gaussian time distribution: sigma={0}, "
                    "time window={1}, time resolution={2}".format(sigma, time_window, time_resolution))
        self.time_resolution = time_resolution
        self.t = np.arange(-time_window/2, time_window, time_resolution)
        self.time_structure = np.exp(-self.t**2 / sigma**2)
        self._generate_t_cdf()

    def set_time_structure(self, time_structure, time_resolution=10e-15):
        logger.info("Setting arbitrary time structure: time resolution={0}".format(time_resolution))
        self.time_resolution = time_resolution
        self.time_structure = time_structure
        self.t = time_resolution * (np.arange(0, time_structure.shape[0]) - time_structure.shape[0]/2)
        self._generate_t_cdf()

    def set_momentum_fermi_dirac(self, E_photon, phi_work=4.31, schottky_field=None):
        """

        :param E_photon: Photon energy in eV (4.7 for 266 nm light)
        :param phi_work: Work function for the metal cathode in eV (4.31 for copper)
        :param schottky_field: Average electric field at cathode at time of emission in MV/m
        :return:
        """
        logger.info("Setting momentum to Fermi dirac distribution: "
                    "E_photon={0}, phi_work={1}, Schottky field={2}".format(E_photon, phi_work, schottky_field))
        me = 9.11e-31
        c = 299792458.0
        qe = 1.602e-19

        if schottky_field is None:
            phi_eff = phi_work
        else:
            phi_eff = phi_work - 0.037947*np.sqrt(schottky_field)
        if phi_eff > E_photon:
            raise ValueError('p_eff must be smaller than E_photon')

        # Isotropic electron momentum distribution inside the crystal
        # Only electrons with angle theta < theta_max will be emitted (refraction)
        # So generate uniform angle distribution in range 0 <= phi < 2*pi, 0 <= theta < theta_max
        n_e = 200
        Ef = 7.0
        E = np.linspace(Ef+phi_eff-E_photon, Ef + E_photon, n_e)
        # E_m = E.reshape((-1, 1)).repeat(n_e, 1)
        theta_max = np.arccos(np.sqrt((Ef+phi_eff)/(E+E_photon)))
        theta_max_m = theta_max.reshape((-1, 1)).repeat(n_e, 1)
        theta = np.linspace(0, theta_max.max(), n_e)
        theta_m = theta.reshape((1, -1)).repeat(n_e, 0)
        self.E0 = Ef+phi_eff #-E_photon
        self.E_max = Ef + E_photon
        self.theta_max = theta_max.max()
        self.theta = theta
        self.E = E
        self.image_fd = (theta_m < theta_max_m).astype(np.double) * np.sin(theta_m)
        # Or should it be
        # self.image_fd = (theta_m < theta_max_m).astype(np.double)

        self.image_fd_cdf = self.image_fd.cumsum(0).cumsum(1)
        self.image_fd_cdf /= self.image_fd_cdf.max()
        # F1 is the normalized 1D cdf in x-direction:
        self.F1_fd = self.image_fd.sum(1).cumsum()
        self.F1_fd /= self.F1_fd.max()
        x = np.arange(self.F1_fd.shape[0])
        self.F1_fd_interp = interp1d(x, self.F1_fd)

        self.Ef = Ef
        self.E_photon = E_photon
        self.phi_eff = phi_eff

    def set_transverse_momentum_gaussian(self, sigma, pixel_resolution, image_size=200):
        """ Generate a gaussian image for use as an electron transverse momentum distribution

        :param sigma: sigma of the gaussian
        :param pixel_resolution: Size of each pixel in the image (eV/c)
        :param image_size: scalar number of pixels in each direction
        :return: nothing
        """
        x = np.arange(-image_size / 2, image_size / 2, dtype=np.int64) * pixel_resolution
        y = np.exp(-(x / np.double(sigma)) ** 2)
        self.momentum_image_pixel_resolution = pixel_resolution
        image = np.outer(y, y)
        self._set_momentum_image(image)

    def set_energy_gaussian(self, sigma, energy_window, energy_resolution=10e-15):
        self.energy_resolution = energy_resolution
        self.energies = np.arange(-energy_window/2, energy_window, energy_resolution)
        self.energy_structure = np.exp(-self.energies**2 / sigma**2)
        self._generate_energy_cdf()

    def set_charge(self, charge):
        logger.info("Setting total beam charge: {0} pC".format(charge*1e12))
        self.charge = charge

    def _generate_x_cdf(self):
        """
        Generate the cumulative distribution function from the stored image.
        :return:
        """
        self.image_cdf = self.image.cumsum(0).cumsum(1)
        self.image_cdf /= self.image_cdf.max()
        # F1 is the normalized 1D cdf in x-direction:
        self.F1 = self.image.sum(1).cumsum()
        self.F1 /= self.F1.max()
        x = np.arange(self.F1.shape[0])
        self.F1_interp = interp1d(x, self.F1)

    def _generate_p_cdf(self):
        """
        Generate the cumulative distribution function from the stored momentum image.
        :return:
        """
        self.momentum_image_cdf = self.momentum_image.cumsum(0).cumsum(1)
        self.momentum_image_cdf /= self.momentum_image_cdf.max()
        # F1 is the normalized 1D cdf in x-direction:
        self.momentum_F1 = self.momentum_image.sum(1).cumsum()
        self.momentum_F1 /= self.momentum_F1.max()
        x = np.arange(self.momentum_F1.shape[0])
        self.momentum_F1_interp = interp1d(x, self.momentum_F1)

    def _generate_t_cdf(self):
        self.t_cdf = self.time_structure.cumsum()
        self.t_cdf /= self.t_cdf.max()
        self.t_int = interp1d(self.t_cdf, self.t, fill_value='extrapolate')

    def _generate_energy_cdf(self):
        self.energy_cdf = self.energy_structure.cumsum()
        self.energy_cdf /= self.energy_cdf.max()
        self.energy_int = interp1d(self.energy_cdf, self.energies, fill_value='extrapolate')

    def _sample_cdf(self, x):
        xi_1 = (self.F1 >= x[0]).searchsorted(True)
        F2 = self.image[xi_1, :].cumsum()
        xi_2 = (F2 / F2.max() >= x[1]).searchsorted(True)
        return np.array([xi_1, xi_2]) * self.image_pixel_resolution

    def sample_x_cdf_interp(self, x_n):
        """
        Sample the 2D cumulative distribution function at the normalized coordinates x[:, 0], x[:, 1].
        On positions between pixels the cdf is linearly interpolated.
        :param x_n: Sample coordinates in range 0..1
        :return: Real coordinates
        """
        # Make sure we have a 2D array even if was 1D (i.e. a single point)
        logging.info('Sample_x_cdf_interp input size {0}'.format(x_n.shape))
        if x_n.ndim < 2:
            x_data = x_n.reshape(1, -1)
        else:
            x_data = x_n
        # Center coordinates:
        x0 = self.image_pixel_resolution * self.image.shape[0] / 2.0
        y0 = self.image_pixel_resolution * self.image.shape[1] / 2.0
        # Loop through particles
        xi = []
        for k in range(x_data.shape[0]):
            xi_high = (self.F1 > x_data[k, 0]).searchsorted(True)
            dfdx = self.F1[xi_high] - self.F1[xi_high - 1]  # dx = 1
            df = self.F1[xi_high] - x_data[k, 0]
            dx = df / dfdx
            xi_1 = xi_high - dx

            F2 = self.image[xi_high, :].cumsum()
            F2 /= F2.max()
            xi_high = (F2 > x_data[k, 1]).searchsorted(True)
            dfdy = F2[xi_high] - F2[xi_high - 1]  # dx = 1
            df = F2[xi_high] - x_data[k, 1]
            dy = df / dfdy
            xi_2 = xi_high - dy
            xi.append(np.array([xi_1, xi_2]))
        p = np.array(xi) * self.image_pixel_resolution
        p[:, 0] = p[:, 0] - x0
        p[:, 1] = p[:, 1] - y0
        return p

    def sample_fermi_dirac_cdf_interp(self, px_n):
        """
        Sample the 3D cumulative distribution function at the normalized coordinates px[:, 0], px[:, 1], px[:, 2].
        On positions between pixels the cdf is linearly interpolated.
        :param px_n: Sample coordinates in range 0..1
        :return: Real coordinates
        """
        # Make sure we have a 2D array even if was 1D (i.e. a single point)
        logging.info('Sample_fermi_dirac_cdf_interp input size {0}'.format(px_n.shape))
        if px_n.ndim < 2:
            px_data = px_n.reshape(1, -1)
        else:
            px_data = px_n
        # Loop through particles
        pxi = []
        me = 9.11e-31
        qe = 1.602e-19
        c = 299792458.0
        for k in range(px_data.shape[0]):
            # First sample energy
            # Find first index where 1d cdf curve is larger than sample:
            pxi_high = (self.F1_fd > px_data[k, 0]).searchsorted(True)
            # Do linear interpolation:
            dfdpx = self.F1_fd[pxi_high] - self.F1_fd[pxi_high - 1]  # dx = 1
            df = self.F1_fd[pxi_high] - px_data[k, 0]
            dpx = df / dfdpx
            pxi_1 = pxi_high - dpx

            # Generate new 1d cdf at the index found:
            eps = 1e-10
            F2_fd = self.image_fd[pxi_high, :].cumsum() + eps
            F2_fd /= F2_fd.max()
            pxi_high = (F2_fd > px_data[k, 1]).searchsorted(True)
            dfdpy = F2_fd[pxi_high] - F2_fd[pxi_high - 1]  # dx = 1
            df = F2_fd[pxi_high] - px_data[k, 1]
            dpy = np.minimum(1.0, df / (dfdpy + eps))
            pxi_2 = np.maximum(0.0, pxi_high - dpy)

            pxi_3 = 2*np.pi*px_data[k, 2]

            # logging.info('E index {0}, theta index {1}'.format(pxi_1, pxi_2))
            E = self.E0 + (self.E_max - self.E0) * pxi_1 / self.image_fd.shape[0]
            theta = self.theta_max * pxi_2 / self.image_fd.shape[1]

            theta_max = np.arccos(np.sqrt((self.Ef+self.phi_eff)/E))
            theta = theta_max * px_data[k, 1]

            # logging.info('E = {0} eV, theta = {1}'.format(E, theta))
            p_tot_out = np.sqrt(2*me*(E-self.Ef-self.phi_eff)*qe)*c/qe
            p_tot_in = np.sqrt(2*me*E*qe)*c/qe
            p_transverse_out = np.sin(theta)*p_tot_in
            # logging.info('p_tot_out = {0}, p_tot_in = {1}'.format(p_tot_out, p_tot_in))
            # logging.info('p_transverse_out {0}'.format(p_transverse_out))
            p_z_out = np.sqrt(p_tot_out**2 - p_transverse_out**2)
            p_x_out = p_transverse_out*np.sin(pxi_3)
            p_y_out = p_transverse_out * np.cos(pxi_3)

            pxi.append(np.array([p_x_out, p_y_out, p_z_out]))

        return np.array(pxi)

    def sample_p_cdf_interp(self, px_n):
        """
        Sample the 2D cumulative momentum distribution function at the
        normalized coordinates px[:, 0], px[:, 1].
        On positions between pixels the cdf is linearly interpolated.
        :param px_n: Sample coordinates in range 0..1
        :return: Real coordinates
        """
        # Make sure we have a 2D array even if was 1D (i.e. a single point)
        if px_n.ndim < 2:
            px_data = px_n.reshape(1, -1)
        else:
            px_data = px_n
        # Loop through particles
        pxi = []
        for k in range(px_data.shape[0]):
            pxi_high = (self.momentum_F1 > px_data[k, 0]).searchsorted(True)
            dfdpx = self.momentum_F1[pxi_high] - self.momentum_F1[pxi_high - 1]  # dx = 1
            df = self.momentum_F1[pxi_high] - px_data[k, 0]
            dpx = df / dfdpx
            pxi_1 = pxi_high - dpx

            F2 = self.momentum_image[pxi_high, :].cumsum()
            F2 /= F2.max()
            pxi_high = (F2 > px_data[k, 1]).searchsorted(True)
            dfdpy = F2[pxi_high] - F2[pxi_high - 1]  # dx = 1
            df = F2[pxi_high] - px_data[k, 1]
            dpy = df / dfdpy
            pxi_2 = pxi_high - dpy
            pxi.append(np.array([pxi_1, pxi_2]))
        return np.array(pxi) * self.momentum_image_pixel_resolution

    def sample_t_cdf_interp(self, t_n):
        """
        Sample the cumulative distribution function for the time structure.
        Interpolates between points in the time structure vector.

        :param t_n: Normalized sample coordinate 0..1
        :return: Corresponding time from time structure
        """
        return self.t_int(t_n)

    def sample_energy_cdf_interp(self, energy_n):
        """
        Sample the cumulative distribution function for the energy structure.
        Interpolates between points in the energy structure vector.

        :param energy_n: Normalized sample coordinate 0..1
        :return: Corresponding energy from energy structure
        """
        return self.energy_int(energy_n)

    def generate_halton(self, n_points, n_dim=2, seed=None):
        """
        Genrate Halton sequence using the ghalton package.
        If the seed value is not None, the sequencer is seeded and reset.

        :param n_points: Number of points to generate
        :param n_dim: Number of dimensions in the generated sequence
        :param seed: Seed value for the generator. See ghalton documentation
        for details. Use seed=-1 for optimized DeRainville2012 sequence permutations.
        :return: Numpy array of shape [n_points, n_dim]
        """
        try:
            p0 = self.halton_sequencer.get(1)
        except (NameError, AttributeError):
            self.halton_sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:n_dim])
        if seed is not None:
            if seed == -1:
                seed = ghalton.EA_PERMS
            self.halton_sequencer.seed(seed)
        p = np.array(self.halton_sequencer.get(n_points))
        return p

    def generate_hammersley(self, n_points, start=0,  n_dim=2, step=1):
        """
        Generate numbers in the quasi random hammersley sequence
        :param n_points: number of points to generate
        :param start: starting point in the sequence
        :param n_dim: number of dimensions to generate in
        :param skip: use every skip:th number in the sequence
        :return: n_dim x n_points array of quasi random numbers
        """
        result = []
        for d in range(n_dim):
            tmp_res = []
            for k in range(start, start + step * n_points, step):
                point = 0
                f = 1.0 / self.saved_primes[d]
                i = k
                while i > 0:
                    point += f * (i % self.saved_primes[d])
                    i = np.floor(i / self.saved_primes[d])
                    f = f / self.saved_primes[d]
                tmp_res.append(point)
            result.append(np.array(tmp_res))
        return np.array(result)

    def generate_hammersley_v(self, n_points, start=0, n_dim=2):
        """
        Generate numbers in the quasi random hammersley sequence
        :param n_points: number of points to generate
        :param start: starting point in the sequence
        :param n_dim: number of dimensions to generate in
        :return: n_dim x n_points array of quasi random numbers
        """
        k = np.transpose(np.repeat(np.arange(start, start+n_points), n_dim).reshape(n_points, n_dim))
        points = np.zeros_like(k, dtype=np.double)
        f = np.ones_like(k, dtype=np.double)
        base = self.saved_primes[0:n_dim].repeat(n_points).reshape(n_dim, n_points)
        f /= base
        alldone = False
        good_ind = np.ones_like(k)
        while alldone is False:
            tmp = f * (k % base)
            points[good_ind] += tmp[good_ind]
            k = np.floor(k / base)
            f = f / base
            good_ind = k > 0
            if good_ind.sum() == 0:
                alldone = True
        return points

    def generate_quasi_random(self, n_points, n_dim=2):
        return np.transpose(lhs(n_dim, samples=n_points))

    def set_random_generator(self, gen='hammersley'):
        """
        Select the (quasi) random number generator to use when generating the particles
        :param gen: 'hammersley', 'random', 'qr'
        :return:
        """
        if gen in ['hammersley', 'random', 'qr']:
            self.gen = gen
        else:
            raise ValueError('gen must be hammersley, random, or qr')

    def generate_particles(self, n):
        """
        Generate n particles over the density map. Coordinates are randomly selected along with
        a probability. If the density in the image at the coordinates is higher than the probability
        the coordinates are selected as a particle. Note that this could be inefficient if the image
        is mostly empty.
        :param n: number of particles to generate
        :return: 2D numpy array of particle coordinates (size n x 2)
        """
        n_rem = n
        particles = np.zeros((2, 1))
        start = 0
        while n_rem > 0:
            if self.gen == 'hammersley':
                random_vec = self.generate_hammersley(n_rem, start, 2)
            elif self.gen == 'qr':
                random_vec = self.generate_quasi_random(n_rem, 2)
            else:
                random_vec = np.random.random((2, n_rem))
            x = random_vec[0, :] * self.image.shape[0]
            y = random_vec[1, :] * self.image.shape[1]
            pr = np.random.random(n_rem)
            good_ind = self.image[x.astype(np.int), y.astype(np.int)] > pr
            particles_tmp = np.array((x[good_ind], y[good_ind]))
            start += n_rem
            n_rem -= particles_tmp.shape[1]
            particles = np.hstack((particles, particles_tmp))
            print start
        return particles[:, 1:]

    def generate_particles_6d_reduction(self, n_particles):
        n_rem = n_particles
        particles = np.zeros((6, 1))
        start = 0
        while n_rem > 0:
            if self.gen == 'hammersley':
                random_vec = self.generate_hammersley(n_rem, start, 6)
            elif self.gen == 'qr':
                random_vec = self.generate_quasi_random(n_rem, 6)
            else:
                random_vec = np.random.random((6, n_rem))
            x = random_vec[0, :] * self.image.shape[0]
            y = random_vec[1, :] * self.image.shape[1]
            t = random_vec[2, :] * (self.t.max() - self.t_min())
            # Total density is the product of the density in independent dimensions:
            point_density = self.image[x.astype(np.int), y.astype(np.int)] * self.t_int[t]
            pr = np.random.random(n_rem)
            good_ind = point_density > pr
            particles_tmp = np.array((x[good_ind], y[good_ind]))
            start += n_rem
            n_rem -= particles_tmp.shape[1]
            particles = np.hstack((particles, particles_tmp))
            print start
        return particles[:, 1:]

    def generate_particles_6d_cdf(self, n_particles):
        """
        Generate n_particles worth of 6D distribution using cumulative density functions for transverse spatial,
        time, transverse momentum, and energy. These cdf:s are created when set_xxx functions are called.
        
        :param n_particles: Number of particles to generate
        :return: Particle distribution as np.array of size [n_particles, 6]
        """
        n_rem = n_particles
        particles = np.zeros((6, 1))
        start = 0
        if self.gen == 'hammersley':
            logging.info('Using Hammersley quasi random distribution')
            random_vec = self.generate_hammersley(n_rem, start, 6, step=7).transpose()
        elif self.gen == 'halton':
            logging.info('Using Halton quasi random distribution')
            random_vec = self.generate_halton(n_rem, 6)
        elif self.gen == 'qr':
            random_vec = self.generate_quasi_random(n_rem, 6)
        else:
            logging.info('Using pseudo random distribution')
            random_vec = np.random.random((6, n_rem))
        logging.info('Random vec generated, shape {0}'.format(random_vec.shape))
        # Generate transverse profile from image
        x = self.sample_x_cdf_interp(random_vec[:, 0:2])
        logging.info('Transverse points generated')

        t = self.sample_t_cdf_interp(random_vec[:, 2]).reshape(-1, 1)
        logging.info('Time points generated')

        # px = self.sample_p_cdf_interp(random_vec[:, 3:5])
        # logging.info('Transverse momentum points generated')
        #
        # energy = self.sample_energy_cdf_interp(random_vec[:, 5]).reshape(-1, 1)
        # logging.info('Energy points generated')

        p = self.sample_fermi_dirac_cdf_interp(random_vec[:, 3:6])
        logging.info('Fermi-Dirac points generated')

        # particles = np.hstack((x, t, px, energy))
        particles = np.hstack((x, t, p))
        return particles

    def save_astra_distribution(self, filename, n_part):
        logger.info("Saving {0} particle distribution to {1}".format(n_part, filename))
        p = self.generate_particles_6d_cdf(n_part)
        p_astra = np.zeros((n_part, 10))
        p_astra[:, 0:2] = p[:, 0:2]         # Transverse data. z=0 for particles generated at cathode.
        p_astra[:, 3:6] = p[:, 3:6]         # Momentum data
        p_astra[:, 6] = p[:, 2] * 1e9       # Time data in ns
        p_astra[:, 7] = self.charge / n_part * 1e9    # Charge per particle in nC
        p_astra[:, 8] = 1                   # Particle index: electrons
        p_astra[:, 9] = -1                  # Status flag: standard particle
        # Reference particle:
        p_astra[0, 0:3] = np.array([0.0, 0.0, 0.0])
        p_astra[0, 3:6] = np.array([0.0, 0.0, 0.0])
        p_astra[0, 6] = 0.0
        p_astra[0, 7] = self.charge / n_part
        p_astra[0, 8] = 1
        p_astra[0, 9] = -1

        fs = '%1.12E  %1.12E  %1.12E  %1.12E  %1.12E  %1.12E  %1.12E  %1.12E  %d  %d'
        np.savetxt(filename, p_astra, fmt=fs)
        logger.debug("Distribution saved.")


if __name__ == '__main__':
    dm = DensityMapNaive()
    dm.set_transverse_gaussian(1.0e-3, 20e-6)
    dm.set_momentum_fermi_dirac(4.71, 4.46, 0)
    n = 200000
    t0 = time.time()
    dm.gen = 'halton'
    p = dm.generate_particles_6d_cdf(n)
    logging.info('Generated {0} particles in {1} s'.format(n, time.time() - t0))
    dm.save_astra_distribution("gaussian_gen.ini", n)

    image = imread("diffuser_0p15deg_500mm_lens_1-1imaging_iris_closed_cal18p5um_0_crop.png")[:, :, 0]
    dm.set_transverse_image(image, 18.5e-6)
    pi = dm.generate_particles_6d_cdf(n)
    logging.info('Generated {0} particles in {1} s'.format(n, time.time() - t0))
    dm.save_astra_distribution("image_gen.ini", n)
