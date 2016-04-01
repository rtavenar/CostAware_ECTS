import numpy
import random

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def random_indices(n, n_samples):
    return numpy.array(random.sample([i for i in range(n)], n_samples))


def opposite_indices(ind, n):
    lst = []
    for i in range(n):
        if i not in ind:
            lst.append(i)
    return numpy.array(lst)