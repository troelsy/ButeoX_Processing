import numpy
import sys
import time

numpy.random.seed(0)

sys.path.append("..")
from utils import fir, save, show

image = numpy.empty((250, 250), dtype=numpy.int16)
image[:] = 5000

# Grid (assumes square image shape)
for n in range(25, image.shape[0], 25):
    image[n - 1, :] = 12000
    image[n + 0, :] = 12000
    image[n + 1, :] = 12000
    image[:, n - 1] = 12000
    image[:, n + 0] = 12000
    image[:, n + 1] = 12000

image[75:175, 75:175] = 22000

image_original = image.copy()

noise_vari = 5000
# noise_passive = 10000
background_noise = (numpy.random.rand(*image.shape) - 0.5) * noise_vari
image += background_noise.astype(numpy.int16)

save(image, "square_grid_noise.png")
save(image_original, "square_grid.png")

func = numpy.median
for xfilters in range(1, 8):
    for terms in [1, 3, 5, 7, 9, 11, 13, 15]:
        t0 = time.time()
        if terms == 1:
            out = image
        else:
            out = fir(image, func, terms, recursions=xfilters)

        # Cast to float64 to avoid int16 overflows
        a = image_original.astype(numpy.float64)
        b = out.astype(numpy.float64)
        mse = numpy.mean((a - b)**2)
        mape = (100 / (image.shape[0] * image.shape[1])) * numpy.sum(numpy.abs((a - b) / a))
        t1 = time.time()
        print("%i-%i\t%f\t%f\t%f" % (terms, xfilters, t1 - t0, mse, mape))

        # if terms == 3 and i == 5:
        #     show(out, title="FIR median; terms: %i" % terms)
