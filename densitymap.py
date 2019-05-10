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


class DensityMap(object):
    def __init__(self, image=None):
        self.image = None
        self.image_pixel_resolution = 30e-6
        self.size = None
        # self.F1 = None
        # self.F1_interp = None
        # self.image_cdf = None
        self.gen = 'halton'
        if image is not None:
            self._set_image(image)

        self.time_structure = None
        self.time_resolution = 10e-15
        self.t = None   # Variables for charge distribution in time
        self.t_int = None
        self.t_cdf = None

        self.momentum_source = "image"
        self.E_photon = 4.7
        self.E_work = 4.41
        self.E_F = 7.0
        self.momentum_image_pixel_resolution = None
        self.momentum_image = None
        self.momentum_image_pixel_resolution = 30e-6
        self.momentum_size = None

        self.energy_structure = None
        self.energy_resolution = 10e-15
        self.energies = None   # Variables for charge distribution in time
        self.energy_int = None
        self.energy_cdf = None
        self.charge = 100e-12

        # Set a reasonable initial distribution:
        self.set_time_uniform(6e-12, 60e-15)
        self.set_transverse_gaussian(1.0e-3, 20e-5, 200)
        self.set_transverse_momentum_gaussian(50, 1, 200)
        self.set_energy_gaussian(0.2, 0.5, 1.0, 0.001)
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

        :param radius: radius of the top hat
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
        """
        Set the transverse density distribution to an image.
        :param image: 2D transverse density distribution
        :param pixel_resolution: Distance between pixels in m
        :return:
        """
        logger.info("Setting transverse image: resolution={0}".format(pixel_resolution))
        self.image_pixel_resolution = pixel_resolution
        self._set_image(image)

    def set_time_uniform(self, time_duration, time_resolution, time_rise=None):
        """
        Set the time structure to a uniform distribution.
        :param time_duration: Duration of the uniform distribution in s
        :param time_resolution:
        :param time_rise: Rise (and fall-) time of the distribution in s. If none, t_rise = time_duration/10
        :return:
        """
        logger.info("Setting uniform time distribution: duration={0}, time resolution={1}".format(time_duration,
                                                                                                  time_resolution))

        if time_rise is None:
            t_rise = time_duration / 10
        else:
            t_rise = time_rise
        nt_rise = np.int(t_rise / time_resolution)
        nt = np.int(1.0 * time_duration / time_resolution + 2 * nt_rise)
        self.time_resolution = time_resolution
        self.t = time_resolution * (np.arange(nt) - np.double(nt) / 2)
        self.time_structure = np.ones_like(self.t)
        self.time_structure[0:nt_rise] = np.linspace(0, 1, nt_rise)
        self.time_structure[nt-nt_rise:] = np.linspace(1, 0, nt_rise)
        self._generate_t_cdf()

    def set_time_gaussian(self, sigma, time_window, time_resolution=10e-15):
        """
        Set the time structure to a gaussian distribution

        :param sigma: Gaussian sigma in s
        :param time_window: Total time window of the generated time distribution in s
        :param time_resolution:
        :return:
        """
        logger.info("Setting gaussian time distribution: sigma={0}, "
                    "time window={1}, time resolution={2}".format(sigma, time_window, time_resolution))
        self.time_resolution = time_resolution
        self.t = np.arange(-time_window/2, time_window, time_resolution)
        self.time_structure = np.exp(-self.t**2 / sigma**2)
        self._generate_t_cdf()

    def set_time_structure(self, time_structure, time_resolution=10e-15):
        """
        Set time structure as an array of the density function.
        :param time_structure: Time density function array
        :param time_resolution: Time interval between points in the time structure in seconds
        :return:
        """
        logger.info("Setting arbitrary time structure: time resolution={0}".format(time_resolution))
        self.time_resolution = time_resolution
        self.time_structure = time_structure
        self.t = time_resolution * (np.arange(0, time_structure.shape[0]) - time_structure.shape[0]/2)
        self._generate_t_cdf()

    def set_momentum_fermi_dirac(self, E_photon, phi_work=4.31, schottky_field=None, E_F=7.0):
        """

        :param E_photon: Photon energy in eV (4.7 for 266 nm light)
        :param phi_work: Work function for the metal cathode in eV (4.31 for copper)
        :param schottky_field: Average electric field at cathode at time of emission in MV/m
        :param E_F: Fermi energy of the cathode material in eV
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

        self.E_photon = E_photon
        self.E_work = phi_eff
        self.E_F = E_F
        self.momentum_source = "fd"

    def set_transverse_momentum_image(self, image, pixel_resolution):
        logger.info("Setting transverse momentum image: resolution={0}".format(pixel_resolution))
        self.image_pixel_resolution = pixel_resolution
        self._set_momentum_image(image)

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

    def set_energy_gaussian(self, sigma, energy_center, energy_window, energy_resolution=0.1):
        """
        Generate a gaussian energy distribution

        :param sigma: Sigma of the generated energy distribution in eV
        :param energy_center:
        :param energy_window: Total energy window of the generated distribution in eV
        :param energy_resolution:
        :return:
        """
        self.energy_resolution = energy_resolution
        self.energies = np.arange(np.maximum(0, energy_center-energy_window/2), energy_center + energy_window / 2,
                                  energy_resolution)
        self.energy_structure = np.exp(-self.energies**2 / sigma**2)
        self._generate_energy_cdf()

    def set_energy_structure(self, energy_structure, energy_resolution=0.01):
        """
        Set energy structure as an array of the density function.

        :param energy_structure: Energy density function array
        :param energy_resolution: Energy interval between points in the structure in eV
        :return:
        """
        logger.info("Setting arbitrary energy structure: energy resolution={0}".format(energy_resolution))
        self.energy_resolution = energy_resolution
        self.energy_structure = energy_structure
        self.energies = energy_resolution * (np.arange(0, energy_structure.shape[0]) - energy_structure.shape[0]/2)
        self._generate_energy_cdf()

    def set_charge(self, charge):
        """
        Set total bunch charge for use when generating Astra particle file
        :param charge: Total charge in C
        :return:
        """
        logger.info("Setting total beam charge: {0} pC".format(charge*1e12))
        self.charge = charge

    def sample_x(self, x_n):
        """
        Sample the 2D density distribution function at the normalized coordinates x[:, 0], x[:, 1].
        On positions between pixels the density function is linearly interpolated.
        :param x_n: Sample coordinates in range 0..1
        :return: Real coordinates
        """
        pdf = self.image
        x0 = self.image_pixel_resolution * self.image.shape[0] / 2.0
        xlim = [-x0, x0]
        y0 = self.image_pixel_resolution * self.image.shape[1] / 2.0
        ylim = [-y0, y0]
        # Create CDF in axis 0 direction by summing in axis 1, then cumsum:
        F = pdf.sum(1).cumsum()
        F /= F.max()

        x = np.interp(x_n[:, 0], F, np.arange(F.shape[0]))
        xi = np.around(x).astype(np.int)  # For indexing

        F2 = pdf.cumsum(axis=1)
        F2 /= F2.max(axis=1).reshape((-1, 1)).repeat(F2.shape[1], axis=1)

        yi = np.greater(F2[xi, :], x_n[:, 1].reshape((-1, 1))).argmax(axis=1)
        y = yi - (F2[xi, yi] - x_n[:, 1]) / (F2[xi, yi] - F2[xi, yi - 1])  # Interpolation

        px = xlim[0] + x * (xlim[1] - xlim[0]) / pdf.shape[0]
        py = ylim[0] + y * (ylim[1] - ylim[0]) / pdf.shape[1]
        p = np.hstack((px.reshape((-1, 1)), py.reshape((-1, 1))))

        return p

    def sample_fermi_dirac(self, random_vec):
        """
        Sample the 3D cumulative distribution function at the normalized coordinates random_vec[:, 0],
        random_vec[:, 1], random_vec[:, 2].
        On positions between pixels the density function is linearly interpolated.
        :param random_vec: Sample coordinates in range 0..1: Energy, cos(theta), phi
        :return: Real coordinates
        """
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
        self.f_fd = f

        F = f.sum(1).cumsum()
        F /= F.max()

        x = np.interp(random_vec[:, 0], F, np.arange(F.shape[0]))
        E = E.min() + x * (E.max() - E.min()) / f.shape[0]

        costh = np.sqrt(E.min() / E) + random_vec[:, 1] * (1 - np.sqrt(E.min() / E))
        p_par = np.sqrt(2 * me * E * qe) * np.sqrt(1 - costh ** 2) * c / qe
        p_z = np.sqrt(2 * me * (E * costh ** 2 - E.min()) * qe) * c / qe

        phi = random_vec[:, 2] * 2 * np.pi
        p_x = p_par * np.cos(phi)
        p_y = p_par * np.sin(phi)

        p = np.hstack((p_x.reshape((-1, 1)), p_y.reshape((-1, 1)), p_z.reshape((-1, 1))))

        return p

    def sample_p(self, p_n):
        """
        Sample the 2D density distribution function at the normalized coordinates x[:, 0], x[:, 1].
        On positions between pixels the density function is linearly interpolated.
        :param p_n: Sample coordinates in range 0..1
        :return: Real coordinates in units of eV/c
        """
        # Make sure we have a 2D array even if was 1D (i.e. a single point)
        if p_n.ndim < 2:
            p_n = p_n.reshape(1, -1)
        pdf = self.momentum_image
        x0 = self.momentum_image_pixel_resolution * pdf.shape[0] / 2.0
        xlim = [-x0, x0]
        y0 = self.momentum_image_pixel_resolution * pdf.shape[1] / 2.0
        ylim = [-y0, y0]
        # Create CDF in axis 0 direction by summing in axis 1, then cumsum:
        F = pdf.sum(1).cumsum()
        F /= F.max()

        x = np.interp(p_n[:, 0], F, np.arange(F.shape[0]))
        xi = np.around(x).astype(np.int)  # For indexing

        F2 = pdf.cumsum(axis=1)
        F2 /= F2.max(axis=1).reshape((-1, 1)).repeat(F2.shape[1], axis=1)

        yi = np.greater(F2[xi, :], p_n[:, 1].reshape((-1, 1))).argmax(axis=1)
        y = yi - (F2[xi, yi] - p_n[:, 1]) / (F2[xi, yi] - F2[xi, yi - 1])  # Interpolation

        px = xlim[0] + x * (xlim[1] - xlim[0]) / pdf.shape[0]
        py = ylim[0] + y * (ylim[1] - ylim[0]) / pdf.shape[1]
        p = np.hstack((px.reshape((-1, 1)), py.reshape((-1, 1))))

        return p

    def sample_t(self, t_n):
        """
        Sample the cumulative distribution function for the time structure.
        Interpolates between points in the time structure vector.

        :param t_n: Normalized sample coordinate 0..1
        :return: Corresponding time from time structure in s
        """
        return self.t_int(t_n)

    def sample_energy(self, energy_n):
        """
        Sample the cumulative distribution function for the energy structure.
        Interpolates between points in the energy structure vector.

        :param energy_n: Normalized sample coordinate 0..1
        :return: Corresponding energy from energy structure in eV
        """
        E = np.maximum(0, self.energy_int(energy_n))
        return E

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

    def set_momentum_source(self, source="fermi-dirac"):
        try:
            s = source.lower()
        except AttributeError:
            s = "fd"
        if s in ["fd", "fermi-dirac"]:
            self.momentum_source = "fd"
        else:
            self.momentum_source = "image"

    def generate_particles_6d(self, n_particles):
        """
        Generate n_particles worth of 6D distribution using density functions for transverse spatial,
        time, transverse momentum, and energy. These density functions are created when set_xxx functions are called.

        :param n_particles: Number of particles to generate
        :return: Particle distribution as np.array of size [n_particles, 6]
        """
        me = 9.11e-31
        qe = 1.602e-19
        c = 299792458.0

        n_rem = n_particles
        particles = np.zeros((6, 1))
        start = 0
        if self.gen == 'hammersley':
            logging.info('Using Hammersley quasi random distribution')
            random_vec = self._generate_hammersley(n_rem, start, 6, step=7).transpose()
        elif self.gen == 'halton':
            logging.info('Using Halton quasi random distribution')
            random_vec = self._generate_halton(n_rem, 6)
        elif self.gen == 'qr':
            random_vec = self._generate_quasi_random(n_rem, 6)
        else:
            logging.info('Using pseudo random distribution')
            random_vec = np.random.random((6, n_rem))
        logging.info('Random vec generated, shape {0}'.format(random_vec.shape))
        # Generate transverse profile from image
        x = self.sample_x(random_vec[:, 0:2])
        logging.info('Transverse points generated')

        t = self.sample_t(random_vec[:, 2]).reshape(-1, 1)
        logging.info('Time points generated')

        if self.momentum_source == "fd":
            p = self.sample_fermi_dirac(random_vec[:, 3:6])
            logging.info('Fermi-Dirac points generated')
        else:
            pt = self.sample_p(random_vec[:, 3:5])
            E = self.sample_energy(random_vec[:, 5])
            pz = np.sqrt(2 * me * E * qe) * c / qe
            p = np.hstack((pt, pz.reshape((-1, 1))))
            logging.info('Momentum points generated')

        # particles = np.hstack((x, t, px, energy))
        particles = np.hstack((x, t, p))
        particles = particles[~np.isnan(particles[:, 1])]
        return particles

    def save_astra_distribution(self, filename, n_part):
        logger.info("Saving {0} particle distribution to {1}".format(n_part, filename))
        n = np.int(n_part)
        p = self.generate_particles_6d(n)
        p_astra = np.zeros((p.shape[0], 10))
        p_astra[:, 0:2] = p[:, 0:2]         # Transverse data. z=0 for particles generated at cathode.
        p_astra[:, 3:6] = p[:, 3:6]         # Momentum data
        p_astra[:, 6] = p[:, 2] * 1e9       # Time data in ns
        p_astra[:, 7] = self.charge / n * 1e9    # Charge per particle in nC
        p_astra[:, 8] = 1                   # Particle index: electrons
        p_astra[:, 9] = -1                  # Status flag: standard particle
        # Reference particle:
        p_astra[0, 0:3] = np.array([0.0, 0.0, 0.0])
        p_astra[0, 3:6] = np.array([0.0, 0.0, 0.0])
        p_astra[0, 6] = 0.0
        p_astra[0, 7] = self.charge / n
        p_astra[0, 8] = 1
        p_astra[0, 9] = -1

        fs = '%1.12E  %1.12E  %1.12E  %1.12E  %1.12E  %1.12E  %1.12E  %1.12E  %d  %d'
        np.savetxt(filename, p_astra, fmt=fs)
        logger.debug("Distribution saved.")

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

    def _set_momentum_image(self, image):
        """
        Set the internal density map momentum image. The image will be normalized.

        :param image: 2D numpy array density map
        :return: nothing
        """
        self.momentum_source = "image"
        self.momentum_size = image.shape
        max_p = image.max()
        self.momentum_image = image / max_p
        self._generate_p_cdf()

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

    def _generate_halton(self, n_points, n_dim=2, seed=None):
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
        p = np.array(self.halton_sequencer.get(int(n_points)))
        return p

    def _generate_hammersley(self, n_points, start=0, n_dim=2, step=1):
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

    def _generate_hammersley_v(self, n_points, start=0, n_dim=2):
        """
        Generate numbers in the quasi random hammersley sequence, vectorized version
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

    def _generate_quasi_random(self, n_points, n_dim=2):
        return np.transpose(lhs(n_dim, samples=n_points))


if __name__ == '__main__':
    dm = DensityMap()
    dm.set_transverse_gaussian(1.0e-3, 20e-6)
    dm.set_transverse_momentum_gaussian(50, 1, 200)
    dm.set_energy_gaussian(0.2, 0.5, 1.0, 0.001)
    n = 500e3
    t0 = time.time()
    dm.gen = 'halton'
    p = dm.generate_particles_6d(n)
    logging.info('Generated {0} particles in {1} s'.format(n, time.time() - t0))
    # dm.save_astra_distribution("gaussian_gen.ini", n)

    image = imread("diffuser_0p15deg_500mm_lens_1-1imaging_iris_closed_cal18p5um_0_crop.png")[:, :, 0]
    dm.set_momentum_fermi_dirac(4.71, 4.46, 0)
    dm.set_transverse_image(image, 18.5e-6)
    t0 = time.time()
    pim = dm.generate_particles_6d(n)
    logging.info('Generated {0} particles in {1} s'.format(n, time.time() - t0))
    dm.save_astra_distribution("diffuser_0p15deg_500mm_lens_1-1imaging_iris_closed_cal18p5um_0_crop_500k_particles.ini", n)

    from scipy.signal import medfilt2d
    picvc2 = imread("vc2_190124.png")[270:390, 580:700, 0]
    pic2 = picvc2 * (medfilt2d(picvc2, 3) > 0.01)
    dm.set_transverse_image(pic2, 29e-6)
    dm.save_astra_distribution("vc_190124_250k.ini", 250e3)

    picvc = imread("vc_190509.png")[131:215, 275:355]
    pic = picvc * (medfilt2d(picvc, 3) > 0)
    dm.set_transverse_image(pic, 29e-6)
    dm.save_astra_distribution("vc_190509_250k.ini", 250e3)
