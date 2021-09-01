import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.shufflenet import FeatureExtractor as shuff_fe
from models.vgg import FeatureExtractor as vgg_fe
from dataloader import ATeX
from utils.initialize_model import initialize_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms = transforms.Compose(transforms_list)

dataset = ATeX(split="train", transform=transforms)
atex = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)


model = initialize_model("vgg", 15)

FILE = "outputs/models/vgg/model.pth"

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])

new_model = vgg_fe(model)


model.to(device)
model.eval()

labels_list = []
features_list = []

for inputs, labels, class_names in atex:

    inputs = inputs.to(device)
    labels_list.append(labels.item())
    # labels_list.append(labels.cpu().detach().numpy())
    with torch.no_grad():
        features = new_model(inputs)
        features_list.append(features.cpu().detach().numpy().reshape(-1))


features = np.asarray(features_list)
labels = np.asarray(labels_list)

print(features.shape)
np.savetxt('./outputs/train_vgg_ftrs.txt', features, delimiter=',')
np.savetxt('./outputs/train_vgg_lbls.txt', labels, delimiter=',')
