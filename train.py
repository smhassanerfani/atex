import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchsummary import summary

from dataloader import ATeX, ToHSV
from utils.initialize_model import initialize_model
from utils.engines import train_model
from utils.loss import FocalLoss
from models.drn import ResNet101
from models.repvgg import get_RepVGG_func_by_name

RESTORE_FROM = "./outputs/models_v2/resnet101_imagenet.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        # ToHSV(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ]),
    'val': transforms.Compose([
        # ToHSV(),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ]),
}

dataset = {x: ATeX(split=x, transform=data_transforms[x]) for x in [
    'train', 'val']}
atex = {x: DataLoader(dataset[x], batch_size=128, shuffle=True,
                      drop_last=False) for x in ['train', 'val']}

# class_names = dataset['train'].classes
# print(class_names)

model_name = "squeezenet"

try:
    os.makedirs(os.path.join("./outputs/", model_name))
except FileExistsError:
    pass

model = initialize_model(model_name, num_classes=15, use_pretrained=True)

# model = ResNet101(img_channel=3, num_classes=15)

# repvgg_build_func = get_RepVGG_func_by_name("RepVGG-A0")
# model = repvgg_build_func(deploy=False)

# saved_state_dict = torch.load(RESTORE_FROM)
#
# new_params = model.state_dict().copy()
#
# for key, value in saved_state_dict.items():
#     if (key.split(".")[0] not in ["head", "dsn", "fc"]):
#         # print(key)
#         new_params[key] = value
#
# model.load_state_dict(new_params)
# print(model)

model = model.to(device)

# MODEL INFORMATION
# summary(model, torch.zeros((128, 3, 32, 32)))
# print(model)
# exit()

# base_lr = 10**random.uniform(-3, -6)
base_lr = 1.6E-02

criterion = FocalLoss()
# criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
# optimizer = optim.Adam(model.parameters(), lr=2.5e-4)

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

model = train_model(model, model_name, atex,
                    criterion, optimizer, base_lr, pdlr=False, scheduler=step_lr_scheduler, num_epochs=50)
