import numpy


def mask(image, level):
    mask = numpy.empty_like(image)
    mask[image > level] = 0
    mask[image <= level] = 1
    return mask


def filter_ratio(blob):
    return (1 / 15) < blob.h / blob.w < (15 / 1) and 20 < blob.h < 250 and 20 < blob.w < 250


class Blob():
    def __init__(self, x, y, w=1):
        self.x = x
        self.y = y
        self.w = w
        self.h = 1

        self.edge_upper = numpy.array(list(zip(range(self.x, self.x + self.w + 1),
                                               [self.y] * (self.w + 1))),
                                      dtype=numpy.int32)

        self.edge_middle = numpy.empty((0, 2), dtype=numpy.int32)
        self.edge_lower = None

        self.last_point = (self.x, self.x + self.w)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str((self.x, self.y, self.w, self.h))

    def __add__(self, other):
        assert other.h == 1

        if self.y != other.y:
            self.edge_middle = numpy.append(self.edge_middle, [[other.x, other.y]], axis=0)
            self.edge_middle = numpy.append(self.edge_middle, [[other.x + other.w, other.y]], axis=0)

            self.h = self.h + other.h
        else:
            pass

        right_self = self.x + self.w
        right_other = other.x + other.w

        right_most = max(right_self, right_other)
        left_most = min(self.x, other.x)

        self.x = left_most
        self.y = min(self.y, other.y)
        self.w = right_most - left_most

        # TODO: This will not cover cases with merging lines with same y-coordinate
        # self.edge_lower = numpy.array(list(zip(range(other.x, other.x + other.w),
        #                                        [other.y] * other.w)),
        #                               dtype=numpy.int32)

        return self

    def merge(self, other):
        self.edge_middle = numpy.append(self.edge_middle, other.edge_middle, axis=0)

        left = min(self.x, other.x)
        right = max(self.x + self.w, other.x + other.w)

        # TODO: This will not cover cases with merging lines with same y-coordinate
        # self.edge_lower = numpy.array(list(zip(range(left, right),
        #                                        [max(self.y, other.y)] * (right - left))),
        #                               dtype=numpy.int32)

        self.x = left
        self.y = min(self.y, other.y)
        self.w = right - left
        self.h = self.h + other.h if self.h != other.h else self.h

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

    # This is stupid, but it works ... slow! Compare all blobs and merge the ones close
    while True:
        go = False
        for a in blobs:
            for b in blobs:
                if a == b:
                    continue

                if a.dist(b) < distance_threshold:
                    a.merge(b)
                    blobs.remove(b)

                    go = True
                    break
            if go:
                break

        if not go:
            break

    return list(filter(lambda x: x.h != 1, blobs))


def find_blobs(input_image, terms=3, multiplier=0.6):
    image = fir(input_image.copy(), numpy.median, terms)

    background_level = numpy.mean(image[:20]) * multiplier
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