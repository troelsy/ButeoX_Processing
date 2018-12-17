import pickle
import numpy
import sys

sys.path.append("..")
from utils import black, mask, interpolate, fir, show, blob_detection, draw_blobs, filter_ratio

with open("multiple.pickle", "rb") as f:
    image = pickle.load(f, encoding="latin1").astype(numpy.float64)

white = numpy.mean(image[:30], axis=0)
image = (image - black) / (white - black)
image = interpolate(image, 3.0, mode="spline")
image = fir(image, numpy.median, 5)

background_level = numpy.mean(image[:20]) * 0.6
image_mask = mask(image, background_level)

blobs = blob_detection(image_mask, 5.0)

print(blobs)
# draw_blobs(image_mask, blobs, edge=False)
# show(image_mask, r=None)


for blob in filter(filter_ratio, blobs):
    cropped = image[blob.y:blob.y + blob.h, blob.x:blob.x + blob.w].copy()
    cropped *= blob.mask
    show(cropped, r=None)
