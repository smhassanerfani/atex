import pickle
import numpy as np
from dataloader import ATeX
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.ae import AELinear
from utils.visualize import savegif

git dataset = ATeX()
features = np.loadtxt('./outputs/train_tsne2d_shufflenet_ftrs.txt', delimiter=',')
labels = np.loadtxt('./outputs/train_tsne2d_shufflenet_lbls.txt', delimiter=',')

features = torch.from_numpy(features.astype(np.float32))
features = DataLoader(features, batch_size=64, shuffle=False, drop_last=False)


model = AELinear()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)

model.to(device)
model.train()

num_epochs = 300
ftrs_per_epoch = []

for epoch in range(num_epochs):
    ftrs_list = []
    for idx, feature in enumerate(features):
        feature = feature.to(device)
        recon, ftrs = model(feature)
        loss = criterion(recon, feature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if idx == 0:
            ftrs_list = ftrs.cpu().detach().numpy()
            continue
        ftrs_list = np.vstack((ftrs_list, ftrs.cpu().detach().numpy()))
 
    scheduler.step()
    ftrs_list = np.asarray(ftrs_list)
 
    ftrs_per_epoch.append(ftrs_list)
    print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

save_path = "./outputs/ae-lin8/model.pth"
torch.save(model.state_dict(), save_path)

Y_seq = np.array(ftrs_per_epoch)
 
with open('outputs/ae-lin8/atex_train.pkl', 'wb') as f:
    pickle.dump(Y_seq, f)

# with (open("outputs/ae-lin8/atex_train.pkl", "rb")) as openfile:
#     Y_seq = pickle.load(openfile)


print(Y_seq.shape)

lo = Y_seq.min(axis=0).min(axis=0).max()
hi = Y_seq.max(axis=0).max(axis=0).min()
limits = ([lo, hi], [lo, hi])

fig_name = "{dataset_name}-ae".format(dataset_name="ATeX")
fig_path = "./outputs/ae-lin8/{file_name}.gif".format(file_name=fig_name)
savegif(Y_seq, labels, "Linear Auto Encoder", fig_path, dataset.classes, limits=limits)
