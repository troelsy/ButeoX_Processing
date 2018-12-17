import matplotlib.pyplot as plt
import numpy
import seaborn
import sys

sys.path.append("..")
from extractor import images


kvs = numpy.array([40, 50, 60, 70, 80])
mas = numpy.array([0.15, 0.37, 0.59, 0.81, 1.03, 1.25])
l = []
for kv in kvs:
    j = []
    for ma in mas:
        j.append(int(numpy.mean(images[kv][ma][200:400, :380])))

    l.append(j)

noise_map_tray = numpy.array(list(l))


l = []
for kv in kvs:
    j = []
    for ma in mas:
        j.append(int(numpy.mean(images[kv][ma][:20])))

    l.append(j)

noise_map_full = numpy.array(list(l))

xticklabels = list(map(lambda x: "%.2f mA" % x, mas))
yticklabels = list(map(lambda x: "%i kV" % x, kvs))

plt.clf()
plt.title("Heatmap of mean sensor readings through tray")
ax = seaborn.heatmap(noise_map_tray,
                     annot=True,
                     fmt="d",
                     linewidth=0.5,
                     xticklabels=xticklabels,
                     yticklabels=yticklabels)
ax.invert_yaxis()
# plt.show()
plt.savefig("heatmap_tray.eps", bbox_inches='tight')

plt.clf()
plt.title("Heatmap of mean sensor readings with full exposure")
ax = seaborn.heatmap(noise_map_full,
                     annot=True,
                     fmt="d",
                     linewidth=0.5,
                     xticklabels=xticklabels,
                     yticklabels=yticklabels)
ax.invert_yaxis()
# plt.show()
plt.savefig("heatmap_full.eps", bbox_inches='tight')


seaborn.set()

# Plot by mA
plt.clf()
plt.title("Exposure value for fixed current with varying voltage")
plt.plot(mas, noise_map_full[4, :], label="80 kV")
plt.plot(mas, noise_map_full[3, :], label="70 kV")
plt.plot(mas, noise_map_full[2, :], label="60 kV")
plt.plot(mas, noise_map_full[1, :], label="50 kV")
plt.plot(mas, noise_map_full[0, :], label="40 kV")
plt.legend()
# plt.show()
plt.savefig("voltage_noise_full.eps", bbox_inches='tight')

plt.clf()
plt.title("Exposure value through sample tray for fixed current with varying voltage")
plt.plot(mas, noise_map_tray[4, :], label="80 kV")
plt.plot(mas, noise_map_tray[3, :], label="70 kV")
plt.plot(mas, noise_map_tray[2, :], label="60 kV")
plt.plot(mas, noise_map_tray[1, :], label="50 kV")
plt.plot(mas, noise_map_tray[0, :], label="40 kV")
plt.legend()
plt.savefig("voltage_noise_tray.eps", bbox_inches='tight')
# plt.show()

# params = numpy.polyfit(kvs, noise_map_full[:, 2], 3)
# print(params)

# def p(x, params):
#     a = params[0]
#     b = params[1]
#     k = params[2]

#     # return a * y**b + k
#     # return a * numpy.power(x, 1) + b * numpy.power(x, 2) + k
#     return a * x + b * numpy.power(x, 2) + k


# fit = p(kvs, params)
# print(fit)

# Plot by kvs
# Note that with a fixed current, the scale is not linear!
plt.clf()
plt.title("Exposure value for fixed voltage with varying current")
seaborn.lineplot(kvs, noise_map_full[:, 5], label="1.25 mA")
seaborn.lineplot(kvs, noise_map_full[:, 4], label="1.03 mA")
seaborn.lineplot(kvs, noise_map_full[:, 3], label="0.81 mA")
seaborn.lineplot(kvs, noise_map_full[:, 2], label="0.59 mA")
seaborn.lineplot(kvs, noise_map_full[:, 1], label="0.37 mA")
seaborn.lineplot(kvs, noise_map_full[:, 0], label="0.15 mA")
plt.savefig("current_noise_full.eps", bbox_inches='tight')
# plt.show()

plt.clf()
plt.title("Exposure value through sample tray for fixed voltage with varying current")
seaborn.lineplot(kvs, noise_map_tray[:, 5], label="1.25 mA")
seaborn.lineplot(kvs, noise_map_tray[:, 4], label="1.03 mA")
seaborn.lineplot(kvs, noise_map_tray[:, 3], label="0.81 mA")
seaborn.lineplot(kvs, noise_map_tray[:, 2], label="0.59 mA")
seaborn.lineplot(kvs, noise_map_tray[:, 1], label="0.37 mA")
seaborn.lineplot(kvs, noise_map_tray[:, 0], label="0.15 mA")
plt.savefig("current_noise_tray.eps", bbox_inches='tight')
# plt.show()
