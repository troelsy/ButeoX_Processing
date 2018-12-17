import matplotlib.pyplot as plt
import pickle
import numpy
import sys

sys.path.append("..")
from utils import fir, draw

boundary = (55, 465)
with open("../test.pickle", "rb") as f:
    image = pickle.load(f, encoding="latin1")
    image = image[:, boundary[0]:boundary[1]]

func = numpy.median
for terms in range(1, 55, 2):
    if terms == 1:
        out = image
    else:
        out = fir(image, func, terms)

    draw(out, title="FIR median; terms: %i" % terms)
