import pickle
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


with open("../test.pickle", "rb") as f:
    image = pickle.load(f, encoding="latin1")


# image = image[220, :]
boundary = (55, 465)
animate = True

if animate:
    for highcut in range(1000, 50000, 100):
        out = butter_bandpass_filter(image, 1, highcut, 100000, order=5)

        print(highcut)

        plt.clf()
        # plt.imshow(out[:, boundary[0]:boundary[1]],
        #            interpolation='none',
        #            extent=[boundary[0], boundary[1], 0, out.shape[0]])
        # plt.plot(range(boundary[0], boundary[1]), image[boundary[0]:boundary[1]], 'r-')
        # plt.plot(range(boundary[0], boundary[1]), out[boundary[0]:boundary[1]], 'b-')
        plt.imshow(out[:, boundary[0]:boundary[1]],
                   vmin=0,
                   vmax=65535,
                   interpolation='none',
                   extent=[boundary[0], boundary[1], 0, out.shape[0]])
        plt.draw()
        plt.pause(.025)

else:
    out = butter_bandpass_filter(image, 1, 10000, 200000, order=5)

    plt.clf()
    plt.plot(range(boundary[0], boundary[1]), image[boundary[0]:boundary[1]], 'r-')
    plt.plot(range(boundary[0], boundary[1]), out[boundary[0]:boundary[1]], 'b-')
    plt.show()
