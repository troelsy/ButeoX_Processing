import pickle
import numpy
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butterworth(image, fs, lowcut, highcut, order):
    image = 65535 - image
    out = butter_bandpass_filter(image, lowcut, highcut, fs, order=order)
    out = (out - 65535) * -1

    return out

# if __name__ == "__main__":
#     fs = 7000.0
#     lowcut = 1.75
#     highcut = 3000.0
#     order = 2

#     with open("../test.pickle", "rb") as f:
#         image = pickle.load(f, encoding="latin1")

#     out = butterworth(image, fs, lowcut, highcut, order)

#     boundary = (55, 465)
#     plt.clf()
#     plt.imshow(out[:, boundary[0]:boundary[1]],
#                vmin=0,
#                vmax=65535,
#                interpolation='none',
#                extent=[boundary[0], boundary[1], 0, out.shape[0]])

#     # plt.savefig("test_%.2f_%.2f_%.2f_%i.eps" % (lowcut, highcut, fs, order), dpi=300,
#     #             bbox_inches='tight')

#     plt.show()


if __name__ == "__main__":
    with open("../test.pickle", "rb") as f:
        image = pickle.load(f, encoding="latin1")

    out = butterworth(image, 4200.0, 500.0, 2000.0, 2)

    boundary = (55, 465)
    plt.clf()
    plt.imshow(out[:, boundary[0]:boundary[1]],
               # vmin=0,
               # vmax=65535,
               interpolation='none',
               extent=[boundary[0], boundary[1], 0, out.shape[0]])

    # plt.savefig("test_%.2f_%.2f_%.2f_%i.eps" % (lowcut, highcut, fs, order), dpi=300,
    #             bbox_inches='tight')

    plt.show()
