import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt


import os
from torchvision import models, transforms
from torchsummary import summary

from torch.utils.data import DataLoader
from dataloader import ATeX
from utils.initialize_model import initialize_model
from utils.engines import train_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms = transforms.Compose(transforms_list)

dataset = {x: ATeX(split=x, transform=transforms) for x in ['train', 'val']}
atex = {x: DataLoader(dataset[x], batch_size=64, shuffle=True,
                      drop_last=False) for x in ['train', 'val']}

dataset_size = {x: len(dataset[x]) for x in ['train', 'val']}
class_names = dataset['train'].classes
# print(class_names)

model_name = "resnet"

try:
    os.makedirs(os.path.join("./outputs/models", model_name))
except FileExistsError:
    pass

model = initialize_model(model_name, num_classes=15, use_pretrained=True)

model = model.to(device)

# MODEL INFORMATION
# summary(model, (3, 32, 32))
# print(model)
# exit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1.0e-2,
                      momentum=0.9, weight_decay=0.0001)

# optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
# step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, model_name, atex,
                    criterion, optimizer, num_epochs=5)
