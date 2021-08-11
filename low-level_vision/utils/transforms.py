
def img_norm(dataset, as_gray=False):
    import numpy as np

    shape = dataset.shape
    dataset = dataset.reshape(dataset.shape[0], -1)
    dataset = dataset.astype(np.float)
    dataset -= np.mean(dataset, axis=0)
    dataset /= np.std(dataset, axis=0)

    if as_gray:

        return dataset.reshape(dataset.shape[0], shape[1], shape[2])

    return dataset.reshape(dataset.shape[0], shape[1], shape[2], shape[3])


def rgb2hsv(dataset):
    from skimage.color import rgb2hsv

    return rgb2hsv(dataset)
