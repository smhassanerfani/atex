import os
import numpy as np
from skimage import io
from tqdm import tqdm
from utils.transforms import img_norm, rgb2hsv


def dataloader(as_gray=False, norm=False, hsv=False, rootdir="/home/serfani/Documents/atex/data/atex"):

    dataset = {
        "classes": [],
        "train": {"data": [], "target": []},
        "test": {"data": [], "target": []},
        "val": {"data": [], "target": []}
    }

    for _set in tqdm(["train", "val", "test"], desc='Loading Progress'):
        datadir = os.path.join(rootdir, _set)
        counter = 0
        for root, dirs, files in os.walk(datadir, topdown=True):
            if counter == 0:
                dataset["classes"] = dirs
            for image in files:
                if image.endswith(".jpg"):
                    dataset[_set]["data"].append(
                        io.imread(os.path.join(root, image), as_gray=as_gray))
                    dataset[_set]["target"].append(counter - 1)
            counter += 1

        dataset[_set]["data"] = np.asarray(dataset[_set]["data"])
        dataset[_set]["target"] = np.asarray(dataset[_set]["target"])

        if norm:
            dataset[_set]["data"] = img_norm(
                dataset[_set]["data"], as_gray=as_gray)
        if hsv:
            dataset[_set]["data"] = rgb2hsv(dataset[_set]["data"])

    return dataset
