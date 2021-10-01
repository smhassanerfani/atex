import pickle
import numpy as np
from dataloader import ATeX
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from utils.engines import adjust_learning_rate
from models.ae import AELinear
from utils.visualize import savegif, plot_2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = ATeX()
features = np.loadtxt('./outputs/train_shufflenet_ftrs.txt', delimiter=',')
labels = np.loadtxt('./outputs/train_shufflenet_lbls.txt', delimiter=',')

features = torch.from_numpy(features.astype(np.float32))
features = DataLoader(features, batch_size=64, shuffle=False, drop_last=False)

model = AELinear()

model.to(device)
model.train()

num_epochs = 300
ftrs_per_epoch = []
in_iter = 0
base_lr = 1.00e-3

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1.00e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)

for epoch in range(num_epochs):
    ftrs_list = []
    tloss = 0.0
    for idx, feature in enumerate(features):
        feature = feature.to(device)
        optimizer.zero_grad()

        recon, ftrs = model(feature)
        loss = criterion(recon, feature)

        # in_iter += feature.shape[0]
        # lr = adjust_learning_rate(
        #     optimizer, base_lr, in_iter, num_epochs * len(features.dataset), 0.9)
        loss.backward()
        optimizer.step()
        tloss += loss.item() * feature.shape[0]

        if idx == 0:
            ftrs_list = ftrs.cpu().detach().numpy()
            continue
        ftrs_list = np.vstack((ftrs_list, ftrs.cpu().detach().numpy()))

    tloss = tloss / len(features.dataset)
    scheduler.step()

    ftrs_list = np.asarray(ftrs_list)
    ftrs_per_epoch.append(ftrs_list)
    print(f"Epoch: {epoch+1}, Loss: {tloss:.6f}")


save_path = "./outputs/ae-lin8/model.pth"
torch.save(model.state_dict(), save_path)

Y_seq = np.array(ftrs_per_epoch)
print(Y_seq.shape)

with open('outputs/ae-lin8/atex_train_v2.pkl', 'wb') as f:
    pickle.dump(Y_seq, f)

# with (open("outputs/ae-lin8/atex_train_v2.pkl", "rb")) as openfile:
#     Y_seq = pickle.load(openfile)

plot_2d(ftrs_list, labels, dataset.classes)

X = Y_seq[:, :, 0]
y = Y_seq[:, :, 1]
limits = ([X.min(), X.max()], [y.min(), y.max()])
print(limits)

fig_name = "{dataset_name}-ae".format(dataset_name="ATeX")
fig_path = "./outputs/ae-lin8/{file_name}.gif".format(file_name=fig_name)
savegif(Y_seq, labels, "Linear Auto Encoder",
        fig_path, dataset.classes, limits=limits)
