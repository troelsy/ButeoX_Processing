import numpy
import warnings
import scipy.misc
import scipy.ndimage
import os
import pickle


with open(os.path.join(os.path.dirname(__file__), "black.pickle"), "rb") as f:
    black = pickle.load(f, encoding="latin1")


def lanczos_transform(image, h_size, v_size):
    return scipy.misc.imresize(image, h_size, v_size, interp="lanczos")


def spline_transform(image, h_ratio, v_ratio):
    with warnings.catch_warnings():
        # Suppress unimportant warning from scipy.ndimage.zoom
        warnings.simplefilter("ignore")
        return scipy.ndimage.zoom(image, (h_ratio, v_ratio), order=5)  # 5 is the highest


def median_transform(image, ratio):
    assert 1 < ratio
    h = image.shape[0]
    h -= h % ratio
    image = image[:h]

    view = [image[n::ratio] for n in range(ratio)]

    return numpy.median(view, axis=0)


def mean_transform(image, ratio):
    assert 1 < ratio
    h = image.shape[0]
    h -= h % ratio
    image = image[:h]

    view = [image[n::ratio] for n in range(ratio)]

    return numpy.mean(view, axis=0)


def gauss(x, sd):
    return 1. / (sd * numpy.sqrt(2 * numpy.pi)) * numpy.exp((-1. / 2.) * numpy.power(x / sd, 2))


def gauss_tranform(image, ratio, sd):
    sample = image[:-(image.shape[0] % ratio)].copy()
    gauss_samples = numpy.arange(-(ratio // 2), ratio % 2 + ratio // 2, 1)
    gaussian_sample = gauss(gauss_samples, sd)
    normalized_gaussian_sample = gaussian_sample / numpy.sum(gaussian_sample)

    view = []
    for n in range(ratio):
        view.append(sample[n::ratio] * normalized_gaussian_sample[n])

    return numpy.sum(view, axis=0)


def interpolate(image, ratio, mode, sd=0.8):
    if mode == "lanczos":
        return lanczos_transform(image, image.shape[0] // ratio, image.shape[1])
    elif mode == "median":
        return median_transform(image, ratio)
    elif mode == "mean":
        return mean_transform(image, ratio)
    elif mode == "gauss":
        return gauss_tranform(image, ratio, sd)
    elif mode == "spline":
        return spline_transform(image, 1. / ratio, 1.)
    else:
        raise Exception("Modes supported: 'median', 'gauss', 'lanczos' and 'spline'")


def create_view(image, h_terms, v_terms=None, padded=False):
    if v_terms is None:
        v_terms = h_terms

    if v_terms % 2 == 0 or v_terms <= 1:
        raise Exception("Terms must be an odd number and greater than 1")
    if h_terms % 2 == 0 or h_terms <= 1:
        raise Exception("Terms must be an odd number and greater than 1")

    h_terms_even = h_terms - 1
    v_terms_even = v_terms - 1
    h_half_term = h_terms_even // 2
    v_half_term = v_terms_even // 2

    if not padded:
        padded_image = numpy.empty((image.shape[0] + h_terms_even, image.shape[1] + v_terms_even))
        padded_image[h_half_term:-h_half_term, v_half_term:-v_half_term] = image
        padded_image[:h_half_term, :] = 0
        padded_image[:, :v_half_term] = 0
        padded_image[-h_half_term:, :] = 0
        padded_image[:, -v_half_term:] = 0
    else:
        padded_image = image

    width, height = padded_image.shape
    view = []
    for x in range(-h_half_term, h_half_term + 1):
        for y in range(-v_half_term, v_half_term + 1):
            view.append(padded_image[h_half_term + x: width - (h_half_term - x),
                                     v_half_term + y: height - (v_half_term - y)])

    return numpy.dstack(view)


def fir(image, func, terms, padded=False, recursions=1):
    return afir(image, func, terms, terms, padded=padded, recursions=recursions)


def afir(image, func, v_terms, h_terms, padded=False, recursions=1):
    view = create_view(image, h_terms, v_terms, padded=padded)

    for n in range(recursions):
        image = func(view, axis=2)

    return image


def calibrate(image):
    white = numpy.mean(image[:30], axis=0)
    image = (image - black) / (white - black)

    return image


def normalize(image, size):
    normalized = numpy.zeros((size, size))

    h, w = image.shape
    if w < h:
        ratio = size / h
        scaled = spline_transform(image, ratio, ratio)
        diff = size - scaled.shape[1]
        half_diff = diff // 2
        mod_diff = diff % 2

        normalized[:, half_diff:-(half_diff + mod_diff)] = scaled
    else:
        ratio = size / w
        scaled = spline_transform(image, ratio, ratio)
        diff = size - scaled.shape[0]
        half_diff = diff // 2
        mod_diff = diff % 2

        normalized[half_diff:-(half_diff + mod_diff), :] = scaled

    return normalized

def adaptive_median(image):
    base = numpy.empty_like(image)

    view0 = create_view(image, 3)

    z_min0 = numpy.min(view0, axis=2)
    z_max0 = numpy.max(view0, axis=2)
    z_med0 = numpy.median(view0, axis=2)
    z_xy0 = view0[:, :, 3 * 3 // 2]

    median_impulse_mask0 = (z_med0 - z_min0 > 0) & (z_med0 - z_max0 < 0)  # If median filter were to output an impulse (1 if not, 0 if impulse)
    xy_impluse_mask0 = (z_xy0 - z_min0 > 0) & (z_xy0 - z_max0 < 0)  # is z_xy an impulse? 1: no, use xy. 0: yes, use median
    xy_impluse_invert_mask0 = numpy.bitwise_not(xy_impluse_mask0)
    level_up_mask0 = numpy.bitwise_not(median_impulse_mask0)  # Found impulse, extend region

    base[median_impulse_mask0 & xy_impluse_mask0] = z_xy0[median_impulse_mask0 & xy_impluse_mask0]
    base[median_impulse_mask0 & xy_impluse_invert_mask0] = z_med0[median_impulse_mask0 & xy_impluse_invert_mask0]

    view1 = create_view(image, 5)

    # This level is different than the first. All views are cropped to the (x,y) in level_up_mask0.
    z_min1 = numpy.min(view1[level_up_mask0], axis=1)
    z_max1 = numpy.max(view1[level_up_mask0], axis=1)
    z_med1 = numpy.median(view1[level_up_mask0], axis=1)
    z_xy1 = view1[level_up_mask0][:, 5 * 5 // 2]

    median_impulse_mask1 = (z_med1 - z_min1 > 0) & (z_med1 - z_max1 < 0)  # If median filter were to output an impulse (1 if not, 0 if impulse)
    xy_impluse_mask1 = (z_xy1 - z_min1 > 0) & (z_xy1 - z_max1 < 0)  # is z_xy an impulse? 1: no, use xy. 0: yes, use median
    level_up_mask1 = numpy.bitwise_not(median_impulse_mask1)  # Found impulse, extend region

    t1 = median_impulse_mask1 * xy_impluse_mask1 * z_xy1
    t2 = median_impulse_mask1 * numpy.bitwise_not(xy_impluse_mask1) * z_med1
    t3 = level_up_mask1 * z_xy1

    base[level_up_mask0] = t1 + t2 + t3

    return base
