import numpy

from utils import show, calibrate, interpolate, find_blobs, normalize
from extractor import images

image = images[40][0.15].astype(numpy.float64).copy()
show(image, r=None)
image = calibrate(image)
show(image, r=None)
image = interpolate(image, 3, mode="gauss")
show(image, r=None)
image = find_blobs(image)[0]
show(image, r=None)
image = normalize(image, 256)
show(image, r=None)
