import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class ATeX(Dataset):
    def __init__(self, rootdir="./dataset/atex", split="train", transform=None):
        super(ATeX, self).__init__()
        self.rootdir = rootdir
        self.split = split
        self.transform = transform
        self.images_base = os.path.join(self.rootdir, self.split)
        self.items_list = self._get_images_list()

    def _get_images_list(self):
        items_list = list()
        for root, dirs, files in os.walk(self.images_base, topdown=True):
            for file in files:
                if file.endswith(".jpg"):
                    items_list.append({
                        "image": os.path.join(root, file),
                        "name": file
                    })
        return items_list

    def __getitem__(self, index):
        pass
