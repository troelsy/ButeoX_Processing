import matplotlib.pyplot as plt
import pickle
import numpy
import sys

sys.path.append("..")
from utils import black, fir, interpolate, save

with open("../chicken.pickle", "rb") as f:
    image_whole = pickle.load(f, encoding="latin1")

with open("../chicken2.pickle", "rb") as f:
    image_broken = pickle.load(f, encoding="latin1")


def prep(image):
    # Convert for easier calculations (raw image is in int16)
    image = image.astype(numpy.float64)

    # Sample max energy level (requires X )
    white = image[:40]
    white = numpy.mean(white, axis=0)

    image = interpolate(image, 3.0, mode="spline")

    # Linear Calibration
    image = (image - black) / (white - black)

    # Scale to original levels
    image = image * 65535

    return image


# fname1 = "case_whole_original.png"
# fname2 = "case_whole_filtered.png"
# image_whole = prep(image_whole)
# image_whole = image_whole[1400:2100][280:430, 120:350]  # Crop image to single piece
# image = image_whole

fname1 = "case_broken_original.png"
fname2 = "case_broken_filtered.png"
image_broken = prep(image_broken)
image_broken = image_broken[1750:2300][20:220, 60:260]  # Crop image to single piece
image = image_broken

save(image, fname1, title="Original Image With No Filter")

func = numpy.median
terms = 3
recursions = 6
image = fir(image, func, terms, recursions=recursions)

save(image, fname2, title="FIR median, %i terms, %i recursions" % (terms, recursions))
