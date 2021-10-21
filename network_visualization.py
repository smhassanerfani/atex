import torch
from utils.initialize_model import initialize_model
from saliency import show_saliency_maps, preprocess
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())

def _get_images_list(path):
    imgs = list()
    lbls = list()
    clss = list()
    for root, dirs, files in os.walk(path, topdown=True):

        for idx, file in enumerate(files):
            if file.endswith(".jpg"):
                image = Image.open(os.path.join(root, file))
                image = preprocess(image)
                imgs.append(image)
                lbls.append(torch.tensor([idx]))
                clss.append(file.split(".")[0])

    return imgs, lbls, clss

images, labels, classes = _get_images_list("outputs/visualization/128x128")

X = torch.cat(images, dim=0).to(device)
y = torch.cat(labels, dim=0).to(device)

model_name = "resnext"
model = initialize_model(model_name, 15)

FILE = f"outputs/models/{model_name}/model.pth"
checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])

for param in model.parameters():
    param.requires_grad = False

model.to(device)

show_saliency_maps(X, y, classes, model, model_name)

