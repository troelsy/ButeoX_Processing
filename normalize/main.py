import numpy
import sys

sys.path.append("..")
from utils import show, calibrate, interpolate, find_blobs, normalize
from extractor import images

image = images[40][0.37].astype(numpy.float64).copy()
image = calibrate(image)
image = interpolate(image, 3, mode="gauss")
image = find_blobs(image)[0]

show(image, r=None)
exit()
image = normalize(image, 256)

show(image, r=None)
