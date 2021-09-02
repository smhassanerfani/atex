import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchsummary import summary

from dataloader import ATeX
from utils.initialize_model import initialize_model
from utils.engines import train_model

from models.drn import ResNet101

RESTORE_FROM = "/home/serfani/Downloads/resnet101_imagenet.pth"
# RESTORE_FROM = "/home/serfani/Downloads/fcn_resnet101_coco-7ecb50ca.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms = transforms.Compose(transforms_list)

dataset = {x: ATeX(split=x, transform=transforms) for x in ['train', 'val']}
atex = {x: DataLoader(dataset[x], batch_size=128, shuffle=True,
                      drop_last=False) for x in ['train', 'val']}

# class_names = dataset['train'].classes
# print(class_names)

model_name = "resnet"

try:
    os.makedirs(os.path.join("./outputs/models", model_name))
except FileExistsError:
    pass

model = initialize_model(model_name, num_classes=15, use_pretrained=True)
# model = ResNet101(img_channel=3, num_classes=15)

# saved_state_dict = torch.load(RESTORE_FROM)
# new_params = model.state_dict().copy()

# for key, value in saved_state_dict.items():
#     if (key.split(".")[0] not in ["head", "dsn", "fc"]):
#         # print(key)
#         new_params[key] = value

# model.load_state_dict(new_params, strict=False)
# print(model)


model = model.to(device)

# MODEL INFORMATION
# summary(model, torch.zeros((64, 3, 32, 32)))
# print(model)
# exit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=2.5e-3,
                      momentum=0.9, weight_decay=0.0001)

# optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
# step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

model = train_model(model, model_name, atex,
                    criterion, optimizer, 2.5e-4, scheduler=None, num_epochs=30)
