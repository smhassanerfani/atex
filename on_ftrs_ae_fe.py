import numpy as np
from dataloader import ATeX
import torch
from torch.utils.data import DataLoader
from models.ae import AELinear
from models.fe import AELinear_FE
from utils.visualize import plot_2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = ATeX()
features = np.loadtxt('./outputs/train_shufflenet_ftrs.txt', delimiter=',')
labels = np.loadtxt('./outputs/train_shufflenet_lbls.txt', delimiter=',')

features = torch.from_numpy(features.astype(np.float32))
features = DataLoader(features, batch_size=64, shuffle=False, drop_last=False)


model = AELinear()

FILE = "outputs/ae-lin8/model.pth"
saved_state_dict = torch.load(FILE)

# new_params = model.state_dict().copy()
# for key, value in saved_state_dict.items():
#     # if key.split(".")[0] not in ["head", "dsn", "fc"]:
#     print(key, value.shape)
#     # new_params[key] = value


model.load_state_dict(saved_state_dict)

new_model = AELinear_FE(model)

new_model.to(device)
new_model.eval()


for idx, inputs in enumerate(features):
    inputs = inputs.to(device)

    with torch.no_grad():
        ftrs = new_model(inputs)
        if idx == 0:
            ftrs_list = ftrs.cpu().detach().numpy()
            continue
        ftrs_list = np.vstack((ftrs_list, ftrs.cpu().detach().numpy()))


features = np.asarray(ftrs_list)
print(features.shape, labels.shape)


# plot_2d(features, labels, dataset.classes)
np.savetxt("./outputs/train_ael8_ftrs.txt", features, delimiter=",")
