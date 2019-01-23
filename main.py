import numpy
import pickle

from utils import show, save, fir, phantom as image


def contra(x, axis=None):
    a = numpy.sum(numpy.power(x, 2), axis=axis)
    b = numpy.sum(x, axis=axis)
    return a / b


def midpoint(x, axis=None):
    a = numpy.max(x, axis=axis)
    b = numpy.min(x, axis=axis)
    return a + b


terms = 3
global_variance = numpy.var(image)

def adaptive_mean(x, axis=None):
    # Something is wrong with this filter
    g_xy = x[:, :, terms * terms // 2]

    if global_variance == 0:
        return g_xy

    local_variance = numpy.var(x, axis=axis)
    local_mean = numpy.mean(x, axis=axis)

    ratio = global_variance / local_variance
    ratio[global_variance > local_variance] = 1.0

    return g_xy - ratio * (g_xy - local_mean)


# image = fir(image, adaptive_mean, terms)
# image = fir(image, numpy.median, terms)
show(image, r=None)
