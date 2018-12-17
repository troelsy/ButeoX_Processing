import matplotlib.pyplot as plt
import numpy
import sys

sys.path.append("..")
from utils import calibrate, interpolate, show, fir, afir
from extractor import images

kv = 40
ma = 0.15
image = images[kv][ma].astype(numpy.float64).copy()
image = calibrate(image)
image = afir(image, numpy.median, 3, 9)
image = interpolate(image, 3, "gauss", sd=0.8)

show(image, r=None)
