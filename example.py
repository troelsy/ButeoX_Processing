import pickle

from utils import show, save, calibrate, interpolate


with open("potato.pickle", "rb") as f:
    image = pickle.load(f, encoding="latin1")

image = calibrate(image)
image = interpolate(image, 3, mode="spline")

show(image, r=None)
