import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.shufflenet import FeatureExtractor as shuff_fe
from dataloader import ATeX
from utils.initialize_model import initialize_model


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