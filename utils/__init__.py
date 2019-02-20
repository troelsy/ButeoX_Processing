import numpy
import itertools
import pickle

numpy.random.seed(0)

from utils.postprocessing import interpolate, create_view, fir, afir, calibrate, normalize, adaptive_median
from utils.plot import show, draw, save
from utils.test_data import phantom, phantom_noise, chicken, test_rect, test_rect_noise
from utils.blobs import mask, draw_blobs, blob_detection, filter_ratio


def mape(subject, truth):
    return 1/subject.size * numpy.sum(numpy.abs((truth - subject) / truth))


def mse(subject, truth):
    return 1/subject.size * numpy.sum(numpy.power(subject - truth, 2))
