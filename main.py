import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.shufflenet import FeatureExtractor as shuff_fe
from dataloader import ATeX
from utils.initialize_model import initialize_model
from utils.plots import plot_2d
from utils.pca import PCA
from utils.lda import LDA
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_list = [transforms.ToTensor(), transforms.Normalize(*mean_std)]
transforms = transforms.Compose(transforms_list)

dataset = ATeX(transform=transforms)
atex = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)


# dataiter = iter(atex)
# images, labels, class_names = dataiter.next()
# print(class_names)

model = initialize_model("shufflenet", 15)

FILE = "outputs/shufflenet_v2_x1_0/model.pth"

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])

new_model = shuff_fe(model)
# print(new_model)

model.to(device)
model.eval()

labels_list = []
features_list = []
classes_list = []

for inputs, labels, class_names in atex:

    if class_names[0] not in classes_list:
        classes_list.append(class_names[0])
    inputs = inputs.to(device)
    labels_list.append(labels.item())
    with torch.no_grad():
        features = new_model(inputs)
        features_list.append(features.cpu().detach().numpy().reshape(-1))


features = np.asarray(features_list)
labels = np.asarray(labels_list)

print(features.shape, labels.shape)
print(classes_list)


lda = LinearDiscriminantAnalysis(n_components=2)
X_r = lda.fit(features, labels).transform(features)

# pca = PCA(2)
# pca.fit(features)
# X_r = pca.transform(features)

plot_2d(X_r, labels, classes_list)
