import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from dataloader import ATeX
from models.ae import AEConv3, AEConv4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
# transforms_list = [transforms.ToTensor()]
transforms = transforms.Compose(transforms_list)

dataset = ATeX(split="val", transform=transforms)
atex = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


base_lr = 1e-3
model = AEConv3()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

model.to(device)

num_epochs = 1000
i_iter = 0
outputs = []
for epoch in range(num_epochs):
    for (img, _, _) in atex:
        i_iter += 256
        img = img.to(device)
        optimizer.zero_grad()
        lr = lr_poly(base_lr, i_iter, num_epochs * len(atex) * 256, 0.9)
        adjust_learning_rate(optimizer, lr)

        recon = model(img)
        loss = criterion(recon, img)

        loss.backward()
        optimizer.step()

    # scheduler.step()
    print(f"Epoch: {epoch+1:3d}, Loss: {loss.item():.4f}")
    if (epoch % 100) == 0:
        outputs.append((epoch, img, recon))


mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

for k in range(len(outputs)):
    plt.figure()
    imgs = outputs[k][1].detach().cpu().numpy()
    recon = outputs[k][2].detach().cpu().numpy()
    for i, item in enumerate(imgs):
        if i >= 20:
            break
        plt.subplot(2, 20, i + 1)
        item = item.transpose((1, 2, 0))
        item = std * item + mean
        plt.axis('off')
        plt.imshow(item)
    for i, item in enumerate(recon):
        if i >= 20:
            break
        plt.subplot(2, 20, 20 + i + 1)
        item = item.transpose((1, 2, 0))
        item = std * item + mean
        plt.axis('off')
        plt.imshow(item)
plt.show()
