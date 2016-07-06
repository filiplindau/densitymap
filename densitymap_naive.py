"""
Created on 06 Jul 2016

@author: Filip Lindau
"""

import numpy as np
from pyDOE import lhs
import time


class DensityMapNaive(object):
    def __init__(self, image=None):
        self.image = None
        self.size = None
        self.gen = 'hammersley'
        if image is not None:
            self.set_image(image)

        self.saved_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
                        139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
                        227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
                        311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                        401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487,
                        491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593,
                        599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659])

    def set_image(self, image):
        """
        Set the internal density map image. The image will be normalized.

        :param image: 2D numpy array density map
        :return: nothing
        """
        self.size = image.shape
        max_p = image.max()
        self.image = image/max_p

    def set_gaussian(self, sigma, size):
        """ Generate a gaussian image for use as an electron distribution

        :param sigma: sigma of the gaussian
        :param size: scalar number of pixels in each direction
        :return: nothing
        """
        x = np.arange(-size/2, size/2, dtype=np.int64)
        y = np.exp(-(x/np.double(sigma))**2)
        image = np.outer(y, y)
        self.set_image(image)

    def set_tophat(self, radius, size):
        """
        Generate a top hat image for use as an electron distribution

        :param radius: radius if the top hat
        :param size: 2 element vector for the image dimensions
        :return:
        """
        image = np.zeros(size)
        x = np.arange(-size[0] / 2, size[0] / 2, dtype=np.int64)
        y = np.arange(-size[1] / 2, size[1] / 2, dtype=np.int64)
        xm, ym = np.meshgrid(x, y)
        good_ind = xm**2 + ym**2 < radius**2
        image[good_ind] = 1
        self.image = image

    def generate_hammersley(self, n_points, start=0,  n_dim=2):
        """
        Generate numbers in the quasi random hammersley sequence
        :param n_points: number of points to generate
        :param start: starting point in the sequence
        :param n_dim: number of dimensions to generate in
        :return: n_dim x n_points array of quasi random numbers
        """
        result = []
        for d in range(n_dim):
            tmp_res = []
            for k in range(start, start+n_points):
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

if __name__ == '__main__':
    dm = DensityMapNaive()
    dm.set_gaussian(20, 100)
    # dm.set_tophat(20, [100, 100])
    # p = dm.generate_particles(10000)