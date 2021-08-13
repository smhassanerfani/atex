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

# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms_list = [transforms.ToTensor()]
transforms = transforms.Compose(transforms_list)

dataset = ATeX(split="val", transform=transforms)
atex = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)


model = AEConv3()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)

model.to(device)

num_epochs = 250
outputs = []
for epoch in range(num_epochs):
    for (img, _, _) in atex:

        img = img.to(device)
        optimizer.zero_grad()

        recon = model(img)
        loss = criterion(recon, img)

        loss.backward()
        optimizer.step()

    scheduler.step()
    print(f"Epoch: {epoch+1:3d}, Loss: {loss.item():.4f}")
    outputs.append((epoch, img, recon))


for k in range(0, num_epochs, 20):
    plt.figure()
    imgs = outputs[k][1].detach().cpu().numpy()
    recon = outputs[k][2].detach().cpu().numpy()
    for i, item in enumerate(imgs):
        if i >= 20:
            break
        plt.subplot(2, 20, i + 1)
        item = item.transpose((1, 2, 0))
        # item = 0.229 * item + 0.485
        plt.axis('off')
        plt.imshow(item)
    for i, item in enumerate(recon):
        if i >= 20:
            break
        plt.subplot(2, 20, 20 + i + 1)
        item = item.transpose((1, 2, 0))
        # item = 0.229 * item + 0.485
        plt.axis('off')
        plt.imshow(item)
plt.show()
