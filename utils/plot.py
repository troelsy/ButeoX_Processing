import matplotlib.pyplot as plt


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


def save(image, name, title=None, r=(0, 65535), dpi=300):
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
    plt.savefig(name, dpi=dpi, bbox_inches='tight')
