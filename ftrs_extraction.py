import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.fe import ShuffleNet_FE, VGG_FE
from dataloader import ATeX
from utils.initialize_model import initialize_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms = transforms.Compose(transforms_list)

dataset = ATeX(split="train", transform=transforms)
atex = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

model_name = "shufflenet"
model = initialize_model(model_name, 15)

FILE = f"outputs/models/{model_name}/model.pth"

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])

if model_name == "shufflenet":
    new_model = ShuffleNet_FE(model)

if model_name == "vgg":
    new_model = VGG_FE(model)


new_model.to(device)
new_model.eval()

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
# np.savetxt('./outputs/train_vgg_ftrs.txt', features, delimiter=',')
# np.savetxt('./outputs/train_vgg_lbls.txt', labels, delimiter=',')
