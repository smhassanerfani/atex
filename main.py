import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataloader import ATeX


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms = transforms.Compose(transforms_list)

dataset = ATeX(transform=transforms)
atex = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

dataiter = iter(atex)
images, labels, class_names = dataiter.next()
print(class_names)
