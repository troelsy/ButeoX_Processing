import numpy
import itertools
import pickle

numpy.random.seed(0)

from utils.postprocessing import interpolate, create_view, fir, afir, calibrate, normalize
from utils.plot import show, draw, save
from utils.test_data import phantom, phantom_noise, chicken, test_rect, test_rect_noise
from utils.blobs import mask, find_blobs, draw_blobs


def mape(subject, truth):
    return 1/len(subject) * numpy.sum(numpy.abs((truth - subject) / truth))


def mse(subject, truth):
    return 1/len(subject) * numpy.sum(numpy.power(subject - truth, 2))
