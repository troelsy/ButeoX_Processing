import pickle

from utils import show, save, calibrate, interpolate, adaptive_median


with open("potato.pickle", "rb") as f:
    image = pickle.load(f, encoding="latin1")

image = calibrate(image)  # Compensate for fixed-pattern noise
image = adaptive_median(image)  # Reduce gaussian noise and remove impulses
image = interpolate(image, 3, mode="median")  # downsample to fit 'normal' y-scale

show(image, r=(0, 1.2))
