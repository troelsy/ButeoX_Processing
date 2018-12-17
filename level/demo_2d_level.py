import pickle
import sys

sys.path.append("..")
from utils import mask, show, draw

with open("../test.pickle", "rb") as f:
    image = pickle.load(f, encoding="latin1")

level = 16000
boundary = (55, 465)
animate = True


if animate:
    for level in range(0, 65535, 500):
        print(level)

        masked = mask(image, level)
        out = image * masked

        draw(out[:, boundary[0]:boundary[1]], title="Level %i/65335" % level, t=0.1)

else:
    mask = mask(image, level)
    out = image * mask

    show(out[:, boundary[0]:boundary[1]])
