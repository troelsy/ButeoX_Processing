import matplotlib.pyplot as plt
import pickle
import numpy


def fir_centerized(vec, terms, func, dtype=None):
    assert terms % 2 == 1

    arr = numpy.empty(len(vec) - terms + 1, dtype=dtype)

    offset = terms // 2
    for n in range(offset, len(vec) - offset):
        arr[n - offset] = func(vec[n - offset:n + offset + 1])

    return arr

boundary = (55, 465)
with open("../test.pickle", "rb") as f:
    image = pickle.load(f, encoding="latin1")
    image = image[:, boundary[0]:boundary[1]]


func = numpy.median
image = image[220, :]

for terms in range(1, 55, 2):
    out = fir_centerized(image, terms, func, dtype=numpy.float64)

    plt.clf()
    plt.plot(range(0, len(image)), image, 'r-')
    plt.plot(range(0, len(out)), out, 'b-')
    plt.draw()
    plt.pause(.5)
