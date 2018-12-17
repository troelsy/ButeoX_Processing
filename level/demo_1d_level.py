import matplotlib.pyplot as plt
import pickle
import sys

sys.path.append("..")
from utils import mask

with open("../test.pickle", "rb") as f:
    image = pickle.load(f, encoding="latin1")


image = image[220, :]
level = 48000
boundary = (55, 465)
animate = True


if animate:
    for level in range(0, 65335, 500):
        masked = mask(image, level)
        out = image * masked

        plt.clf()
        plt.plot(range(boundary[0], boundary[1]), image[boundary[0]:boundary[1]], 'r-')
        plt.plot([boundary[0], boundary[1]], [level, level], 'b-')
        plt.plot(range(boundary[0], boundary[1]), out[boundary[0]:boundary[1]], 'b-')
        plt.draw()
        plt.pause(.025)

else:
    mask = mask(image, level)
    out = image * mask

    plt.clf()
    plt.plot(range(boundary[0], boundary[1]), image[boundary[0]:boundary[1]], 'r-')
    plt.plot([boundary[0], boundary[1]], [level, level], 'b-')
    plt.plot(range(boundary[0], boundary[1]), out[boundary[0]:boundary[1]], 'b-')
    plt.show()
