import numpy as np

def img_norm(dataset, as_gray=False):

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


def kernel3C(kernel):
    arrays = [kernel for _ in range(3)]
    return np.stack(arrays, axis=2)


def power(image, kernel, as_gray=False):
    from scipy import ndimage as ndi

    if as_gray:
        real_feature = ndi.convolve(image, np.real(kernel), mode='wrap')
        imag_feature = ndi.convolve(image, np.imag(kernel), mode='wrap')
    else:
        kernel = kernel3C(kernel)
        real_feature = ndi.convolve(
            image, np.real(kernel), mode='wrap')[:, :, 1]
        imag_feature = ndi.convolve(
            image, np.imag(kernel), mode='wrap')[:, :, 1]

    return np.sqrt(real_feature**2 + imag_feature**2)
