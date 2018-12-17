import matplotlib.pyplot as plt
import numpy
import pickle
import warnings
import scipy.misc
import scipy.ndimage
import os

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
    assert ratio == 3
    h = image.shape[0]
    h -= h % 3
    image = image[:h]

    return numpy.max((image[0::3], image[1::3], image[2::3]), axis=0)


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
    elif mode == "gauss":
        return gauss_tranform(image, ratio, sd)
    elif mode == "spline":
        return spline_transform(image, 1. / ratio, 1.)
    else:
        raise Exception("Modes supported: 'median', 'gauss', 'lanczos' and 'spline'")


def mask(image, level):
    mask = numpy.empty_like(image)
    mask[image > level] = 0
    mask[image <= level] = 1
    return mask


def fir(image, func, terms, padded=False, recursions=1):
    return afir(image, func, terms, terms, padded=padded, recursions=recursions)


def afir(image, func, v_terms, h_terms, padded=False, recursions=1):
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

    for n in range(recursions):
        padded_image[h_half_term:-h_half_term, v_half_term:-v_half_term] = func(numpy.dstack(view),
                                                                                axis=2)

    return padded_image[h_half_term:-h_half_term, v_half_term:-v_half_term]


def show(image, title=None, r=(0, 65535)):
    plt.clf()
    if title is not None:
        plt.title(title)

    if r is not None:
        plt.imshow(image,
                   vmin=r[0],
                   vmax=r[1],
                   interpolation='none',
                   extent=[0, image.shape[1], image.shape[0], 0])
    else:
        plt.imshow(image,
                   interpolation='none',
                   extent=[0, image.shape[1], image.shape[0], 0])
    plt.show()


def draw(image, t=0.25, title=None, r=(0, 65535)):
    plt.clf()
    if title is not None:
        plt.title(title)

    if r is not None:
        plt.imshow(image,
                   vmin=r[0],
                   vmax=r[1],
                   interpolation='none',
                   extent=[0, image.shape[1], 0, image.shape[0]])
    else:
        plt.imshow(image,
                   interpolation='none',
                   extent=[0, image.shape[1], 0, image.shape[0]])
    plt.draw()
    plt.pause(t)


def save(image, name, title=None, r=(0, 65535)):
    plt.clf()
    if title is not None:
        plt.title(title)

    if r is not None:
        plt.imshow(image,
                   vmin=r[0],
                   vmax=r[1],
                   interpolation='none',
                   extent=[0, image.shape[1], 0, image.shape[0]])
    else:
        plt.imshow(image,
                   interpolation='none',
                   extent=[0, image.shape[1], 0, image.shape[0]])
    plt.savefig(name, dpi=300, bbox_inches='tight')


def filter_ratio(blob):
    return (1 / 15) < blob.h / blob.w < (15 / 1) and 20 < blob.h < 250 and 20 < blob.w < 250


class Blob():
    def __init__(self, x, y, w=1):
        self.x = x
        self.y = y
        self.w = w
        self.h = 1

        # self.mask = numpy.array([1] * self.w, dtype=numpy.int32)

        self.edge_upper = numpy.array(list(zip(range(self.x, self.x + self.w + 1),
                                               [self.y] * (self.w + 1))),
                                      dtype=numpy.int32)

        self.edge_middle = numpy.empty((0, 2), dtype=numpy.int32)
        self.edge_lower = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str((self.x, self.y, self.w, self.h))

    def __add__(self, other):
        assert other.h == 1

        right_self = self.x + self.w
        right_other = other.x + other.w

        right_most = max(right_self, right_other)
        left_most = min(self.x, other.x)

        self.x = left_most
        self.y = min(self.y, other.y)
        self.w = right_most - left_most

        # Normally, we are merging lines stacked on top of each other, but sometimes the lines are
        # close enough to merge on the same line
        if self.y != other.y:
            left_side = [[other.x, other.y]]
            right_side = [[other.x + other.w, other.y]]

            self.edge_middle = numpy.append(self.edge_middle, left_side, axis=0)
            self.edge_middle = numpy.append(self.edge_middle, right_side, axis=0)

            self.h = self.h + other.h
        else:
            # TODO: update mask instead of appending
            pass

        # TODO: This will not cover cases with merging lines with same y-coordinate
        self.edge_lower = numpy.array(list(zip(range(other.x, other.x + other.w),
                                               [other.y] * other.w)),
                                      dtype=numpy.int32)

        return self

    @property
    def edges(self):
        if self.edge_lower is None:
            return numpy.vstack((self.edge_upper, self.edge_middle))
        else:
            return numpy.vstack((self.edge_upper, self.edge_middle, self.edge_lower))

    @property
    def mask(self):
        edges = self.edge_middle.T
        image_mask = numpy.zeros((self.h, self.w))

        # Repaint body of mask from edges
        for i in range(0, edges.shape[1], 2):
            relative_y = edges[1, i] - self.y - 1
            relative_x_start = edges[0, i] - self.x
            relative_x_end = edges[0, i + 1] - self.x
            image_mask[relative_y, relative_x_start:relative_x_end] = 1

        # Repaint top row of mask from edges
        max_x, max_y = numpy.max(self.edge_upper, axis=0)
        min_x, min_y = numpy.min(self.edge_upper, axis=0)
        relative_y = min_y - self.y
        relative_x_start = min_x - self.x
        relative_x_end = max_x - self.x
        image_mask[relative_y, relative_x_start:relative_x_end] = 1

        return image_mask

    @property
    def y_bounds(self):
        return (self.y, self.y + self.h)

    @property
    def x_bounds(self):
        return (self.x, self.x + self.w)

    def dist(self, other):
        ds = numpy.linalg.norm(self.edges[:, numpy.newaxis] - other.edges, axis=2)
        return numpy.min(ds)


def near(target, blobs, distance_threshold):
    nearest = []
    for blob in blobs:
        d = target.dist(blob)
        if d <= distance_threshold:
            nearest.append(blob)

    if len(nearest) > 0:
        return nearest[-1]  # TODO: Why does it work with selecting the last???
    else:
        return None


def blob_detection(image, distance_threshold):
    width = image.shape[1]
    line_segments = []
    for row_i, row in enumerate(image):
        line_start = None
        line_end = None
        for col_i, v in enumerate(row):
            if v == 1 and line_start is None:
                line_start = col_i

            if (v == 0 or col_i == width - 1) and line_start is not None:
                line_end = col_i

            if line_start is not None and line_end is not None:
                if line_end - line_start != 0:
                    line_segments.append(Blob(line_start, row_i, line_end - line_start))
                    line_start = None
                    line_end = None
                # else:
                #     print("Skip blob with width = 0")

    blobs = []
    for i, line_segment in enumerate(line_segments):
        closest_blob = near(line_segment, blobs, distance_threshold)

        if closest_blob is None:
            blobs.append(line_segment)
        else:
            closest_blob += line_segment

    return blobs


def find_blobs(input_image, terms=3):
    image = fir(input_image.copy(), numpy.median, terms)

    background_level = numpy.mean(image[:20]) * 0.6
    image_mask = mask(image, background_level)

    blobs = blob_detection(image_mask, 5.0)

    filtered = map(lambda b: image[b.y:b.y + b.h, b.x:b.x + b.w].copy() * b.mask,
                   filter(filter_ratio, blobs))

    return list(filtered)


def draw_blobs(image, blobs, edge=True):
    for blob in blobs:
        for yy in range(*blob.y_bounds):
            for xx in range(*blob.x_bounds):
                if xx % 2 == 0:
                    image[yy, xx] += 1
                else:
                    image[yy, xx] += 2

        if edge:
            for x, y in blob.edge_upper:
                image[y, x] += 5

            for x, y in blob.edge_middle:
                image[y, x] += 5

            if blob.edge_lower is not None:
                for x, y in blob.edge_lower:
                    image[y, x] += 5


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
