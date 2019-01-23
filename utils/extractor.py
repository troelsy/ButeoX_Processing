import matplotlib.pyplot as plt
import pickle
import os

files = ["test-40kv-0.15ma-0.20tint-1000lr-21_11_2018-13_43_19.pickle",
         "test-40kv-0.15ma-0.20tint-1000lr-21_11_2018-13_59_28.pickle",
         "test-40kv-0.15ma-0.20tint-1000lr-21_11_2018-14_04_36.pickle",
         "test-40kv-0.15ma-0.20tint-1000lr-21_11_2018-14_06_39.pickle",
         "test-40kv-0.37ma-0.20tint-1000lr-21_11_2018-14_07_35.pickle",
         "test-40kv-0.59ma-0.20tint-1000lr-21_11_2018-13_43_55.pickle",
         "test-40kv-0.59ma-0.20tint-1000lr-21_11_2018-14_08_26.pickle",
         "test-40kv-0.81ma-0.20tint-1000lr-21_11_2018-14_08_48.pickle",
         "test-40kv-1.03ma-0.20tint-1000lr-21_11_2018-13_44_37.pickle",
         "test-40kv-1.03ma-0.20tint-1000lr-21_11_2018-14_09_56.pickle",
         "test-40kv-1.25ma-0.20tint-1000lr-21_11_2018-14_10_17.pickle",
         "test-40kv-1.25ma-0.20tint-1000lr-21_11_2018-14_10_45.pickle",
         "test-50kv-0.15ma-0.20tint-1000lr-21_11_2018-14_14_39.pickle",
         "test-50kv-0.37ma-0.20tint-1000lr-21_11_2018-14_15_34.pickle",
         "test-50kv-0.59ma-0.20tint-1000lr-21_11_2018-14_16_27.pickle",
         "test-50kv-0.81ma-0.20tint-1000lr-21_11_2018-14_17_17.pickle",
         "test-50kv-1.03ma-0.20tint-1000lr-21_11_2018-14_18_25.pickle",
         "test-50kv-1.25ma-0.20tint-1000lr-21_11_2018-14_19_16.pickle",
         "test-60kv-0.15ma-0.20tint-1000lr-21_11_2018-14_20_08.pickle",
         "test-60kv-0.37ma-0.20tint-1000lr-21_11_2018-14_21_01.pickle",
         "test-60kv-0.59ma-0.20tint-1000lr-21_11_2018-14_22_04.pickle",
         "test-60kv-0.81ma-0.20tint-1000lr-21_11_2018-14_22_49.pickle",
         "test-60kv-1.03ma-0.20tint-1000lr-21_11_2018-14_23_16.pickle",
         "test-60kv-1.25ma-0.20tint-1000lr-21_11_2018-14_24_05.pickle",
         "test-70kv-0.15ma-0.20tint-1000lr-21_11_2018-14_24_57.pickle",
         "test-70kv-0.37ma-0.20tint-1000lr-21_11_2018-14_26_04.pickle",
         "test-70kv-0.59ma-0.20tint-1000lr-21_11_2018-14_28_07.pickle",
         "test-70kv-0.81ma-0.20tint-1000lr-21_11_2018-14_28_35.pickle",
         "test-70kv-1.03ma-0.20tint-1000lr-21_11_2018-14_29_04.pickle",
         "test-70kv-1.25ma-0.20tint-1000lr-21_11_2018-14_29_33.pickle",
         "test-80kv-0.15ma-0.20tint-1000lr-21_11_2018-14_30_24.pickle",
         "test-80kv-0.37ma-0.20tint-1000lr-21_11_2018-14_31_15.pickle",
         "test-80kv-0.59ma-0.20tint-1000lr-21_11_2018-14_32_03.pickle",
         "test-80kv-0.81ma-0.20tint-1000lr-21_11_2018-14_33_51.pickle",
         "test-80kv-1.03ma-0.20tint-1000lr-21_11_2018-14_34_20.pickle",
         "test-80kv-1.25ma-0.20tint-1000lr-21_11_2018-14_42_40.pickle"]

path = "/Users/troelsynddal/Dropbox/Documents/DIKU/Kandidat/Speciale/samples/single-potato-multi-energy"

images = {40: {},
          50: {},
          60: {},
          70: {},
          80: {}}


# from utils import calibrate, interpolate, save

for file in files:
    with open("%s/%s" % (path, file), "rb") as f:
        image = pickle.load(f, encoding="latin1")

    s = file.split("-", 3)
    kv = int(s[1][:2])
    ma = float(s[2][:4])

    images[kv][ma] = image

#     images[kv][ma] = interpolate(calibrate(image), 3, mode="gauss")
#     fname = "%ikv-%.2fma.png" % (kv, ma)

#     if os.path.exists(fname):
#         fname = "%ikv-%.2fma_%s.png" % (kv, ma, file.split("-")[-1].split(".")[0])

#     save(images[kv][ma], fname, title="%ikv, %.2fma" % (kv, ma), r=(0, 1), dpi=800)
