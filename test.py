import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader import ATeX
from utils.initialize_model import initialize_model
from sklearn.metrics import classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms = transforms.Compose(transforms_list)

dataset = ATeX(split='test', transform=transforms)
atex = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

class_names = dataset.classes
# print(class_names)

model_name = "vgg"
model = initialize_model(model_name, num_classes=15, use_pretrained=True)

FILE = f"./outputs/{model_name}/model.pth"

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
epoch = checkpoint['epoch']

model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(len(class_names))]
    n_class_samples = [0 for i in range(len(class_names))]
    for image, label, _ in atex:

        image = image.to(device)
        label = label.to(device)
        output = model(image)

        _, predicted = torch.max(output, 1)
        n_samples += label.size(0)
        n_correct += (predicted == label).sum().item()

        y_true.append(label.item())
        y_pred.append(predicted.item())


final_report = classification_report(y_true, y_pred, target_names=class_names)


print(final_report)

# with open(f".output/models/{model_name}/final_report.txt", "w") as fh:
#     fh.write(final_report)
