import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.shufflenet import FeatureExtractor as shuff_fe
from dataloader import ATeX
from utils.initialize_model import initialize_model
from utils.visualize import plot_2d
from utils.pca import PCA
from utils.lda import LDA
import numpy as np
import pickle

from torch.optim import lr_scheduler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms = transforms.Compose(transforms_list)

dataset = ATeX(transform=transforms)
atex = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)


model = initialize_model("shufflenet", 15)

FILE = "outputs/shufflenet_v2_x1_0/model.pth"

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])

new_model = shuff_fe(model)


model.to(device)
model.eval()

labels_list = []
features_list = []

for inputs, labels, class_names in atex:

    inputs = inputs.to(device)
    # labels_list.append(labels.item())
    labels_list.append(labels.cpu().detach().numpy())
    with torch.no_grad():
        features = new_model(inputs)
        features_list.append(features.cpu().detach().numpy())


features = np.asarray(features_list)
labels = np.asarray(labels_list)

np.savetxt('./outputs/train_tsne2d_shufflenet_ftrs.txt',
           features, delimiter=',')
np.savetxt('./outputs/train_tsne2d_shufflenet_lbls.txt', labels, delimiter=',')

# second phase
features = np.loadtxt(
    './outputs/train_tsne2d_shufflenet_ftrs.txt', delimiter=',')
labels = np.loadtxt(
    './outputs/train_tsne2d_shufflenet_lbls.txt', delimiter=',')

features = features.astype(np.float32)
features = torch.from_numpy(features)

features = DataLoader(features, batch_size=64, shuffle=False, drop_last=False)


# lda = LinearDiscriminantAnalysis(n_components=2)
# X_r = lda.fit(features, labels).transform(features)

# pca = PCA(2)
# pca.fit(features)
# X_r = pca.transform(features)

# plot_2d(X_r, labels, dataset.classes)


from models.ae import AELinear
import torch.nn as nn
import torch.optim as optim


# model = AELinear()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
# model.to(device)

# ftrs_per_epoch = []

# model.train()
# num_epochs = 100
# for epoch in range(num_epochs):
#     ftrs_list = []
#     for idx, feature in enumerate(features):
#         feature = feature.to(device)
#         recon, ftrs = model(feature)
#         loss = criterion(recon, feature)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if idx == 0:
#             ftrs_list = ftrs.cpu().detach().numpy()
#             continue
#         ftrs_list = np.vstack((ftrs_list, ftrs.cpu().detach().numpy()))
#         # ftrs_list.append(ftrs.cpu().detach().numpy())

#     scheduler.step()
#     ftrs_list = np.asarray(ftrs_list)

#     ftrs_per_epoch.append(ftrs_list)
#     print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# # # save_path = "./outputs/ae-lin8/model.pth"
# # # torch.save(model.state_dict(), save_path)


# Y_seq = np.array(ftrs_per_epoch)
# print(Y_seq.shape)


# with open('outputs/ae-lin8/atex_train.pkl', 'wb') as f:
#     pickle.dump(Y_seq, f)


with (open("outputs/ae-lin8/atex_train.pkl", "rb")) as openfile:
    Y_seq = pickle.load(openfile)

print(Y_seq.shape)
print(len(Y_seq))

import matplotlib.pyplot as plt
from utils.visualize import savegif


lo = Y_seq.min(axis=0).min(axis=0).max()
hi = Y_seq.max(axis=0).max(axis=0).min()
limits = ([lo, hi], [lo, hi])
fig_name = '%s-ae' % ("ATeX_")
fig_path = './outputs/ae-lin8/%s.gif' % fig_name
savegif(Y_seq, labels, "AE", fig_path, dataset.classes, limits=limits)

exit()
# cmap = plt.get_cmap('tab20', lut=len(dataset.classes))

# fig, (ax_r0, ax_r1) = plt.subplots(2, 10)


# for ax, ftrs_ in zip(ax_r0, ftrs_per_epoch[:10]):
#     # ax.axis('off')
#     ftrs_ = np.asarray(ftrs_)
#     ax.scatter(ftrs_[:, 0], ftrs_[:, 1], c=labels, cmap=cmap)


# for ax, ftrs_ in zip(ax_r1, ftrs_per_epoch[-10:]):
#     # ax.axis('off')
#     ftrs_ = np.asarray(ftrs_)
#     ax.scatter(ftrs_[:, 0], ftrs_[:, 1], c=labels, cmap=cmap)

# plt.show()

# X_r = np.asarray(ftrs_list)
# plot_2d(X_r, labels, dataset.classes)


from sklearn.manifold import TSNE

from models.tsne import extract_sequence
from utils.visualize import savegif


tsne = TSNE(perplexity=20, learning_rate=200,
            n_iter=300, verbose=2, method="barnes_hut")
tsne._EXPLORATION_N_ITER = 100
Y_seq = extract_sequence(tsne, features)

lo = Y_seq.min(axis=0).min(axis=0).max()
hi = Y_seq.max(axis=0).max(axis=0).min()

limits = ([lo, hi], [lo, hi])
fig_name = '%s-%d-%d-tsne' % ("ATeX", 300, 100)
fig_path = './outputs/%s.gif' % fig_name
savegif(Y_seq, labels, "t-SNE", fig_path, classes, limits=limits)
