from torch.utils.data import Dataset
from PIL import Image
import os


class ToHSV:
    def __call__(self, sample):
        # inputs = sample
        return sample.convert('HSV')


class ATeX(Dataset):
    def __init__(self, rootdir="./data/atex", split="train", as_gray=False, transform=None):
        super(ATeX, self).__init__()
        self.rootdir = rootdir
        self.split = split
        self.transform = transform
        self.images_base = os.path.join(self.rootdir, self.split)
        self.items_list, self.classes = self._get_images_list()
        self.as_gray = as_gray

    def _get_images_list(self):
        items_list = list()
        classes = list()
        counter = 0
        for root, dirs, files in os.walk(self.images_base, topdown=True):
            if counter == 0:
                classes = dirs

            for file in files:
                if file.endswith(".jpg"):
                    items_list.append({
                        "image": os.path.join(root, file),
                        "label": counter - 1,
                        "class_name": classes[counter - 1]
                    })
            counter += 1
        return items_list, classes

    # @property
    # def classes(self):
    #     classes = list()
    #     with os.scandir(self.images_base) as it:
    #         for entry in it:
    #             if entry.is_dir():
    #                 classes.append(entry.name)
    #     return classes

    def __getitem__(self, index):
        image_path = self.items_list[index]["image"]
        label = self.items_list[index]["label"]
        class_name = self.items_list[index]["class_name"]

        if self.as_gray:
            image = Image.open(image_path).convert('L')

        else:
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, class_name

    def __len__(self):
        return len(self.items_list)
