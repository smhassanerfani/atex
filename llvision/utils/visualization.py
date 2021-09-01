import numpy as np
import matplotlib.pyplot as plt
# import matplotlib


def plot_samples(dataset, num_of_samples=7):

    font = {'font.family': 'Times New Roman', 'font.size': 14}
    plt.rcParams.update(**font)

    for y, cls in enumerate(dataset["classes"]):
        idxs = np.flatnonzero(dataset["train"]["target"] == y)
        idxs = np.random.choice(idxs, num_of_samples, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * len(dataset["classes"]) + y + 1
            plt.subplot(num_of_samples, len(dataset["classes"]), plt_idx)
            plt.imshow(dataset["train"]["data"][idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
