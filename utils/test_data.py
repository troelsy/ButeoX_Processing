import numpy
import pickle
import os

from skimage.io import imread
from skimage import data_dir

from utils.postprocessing import interpolate, calibrate

try:
    from utils.extractor import images

    potato037 = images[50][0.37].astype(numpy.float64).copy()
    potato037 = calibrate(potato037)
    potato037 = interpolate(potato037, 3, mode="spline")

    potato125 = images[50][1.25].astype(numpy.float64).copy()
    potato125 = calibrate(potato125)
    potato125 = interpolate(potato125, 3, mode="spline")

    with open(os.path.join(os.path.dirname(__file__), "chicken.pickle"), "rb") as f:
        chicken = pickle.load(f, encoding="latin1")
    chicken = calibrate(chicken)
    chicken = interpolate(chicken, 3, mode="spline")
    chicken = chicken[1400:2100]


except ModuleNotFoundError:
    pass

phantom = imread(data_dir + "/phantom.png", as_gray=True)

norm_noise = numpy.random.normal(0, 0.025, size=phantom.shape[0] * phantom.shape[1]).reshape(phantom.shape)
sp_noise = numpy.random.rand(phantom.shape[0] * phantom.shape[1]).reshape(phantom.shape)
phantom_noise = phantom + norm_noise
phantom_noise[sp_noise < 0.15] = 0
phantom_noise[sp_noise >= 0.85] = 1


test_rect = numpy.empty((250, 250))
test_rect[:] = 0.125
# Grid (assumes square image shape)
for n in range(25, test_rect.shape[0], 25):
    test_rect[n - 1, :] = 0.25
    test_rect[n + 0, :] = 0.25
    test_rect[n + 1, :] = 0.25
    test_rect[:, n - 1] = 0.25
    test_rect[:, n + 0] = 0.25
    test_rect[:, n + 1] = 0.25

test_rect[75:175, 75:175] = 0.4

test_rect_noise = test_rect.copy()

noise_vari = 0.05
background_noise = (numpy.random.rand(*test_rect_noise.shape) - 0.5) * noise_vari
test_rect_noise += background_noise
