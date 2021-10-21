import torch
from utils.initialize_model import initialize_model
from utils.saliency import show_saliency_maps, get_images_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())


images, labels, classes = get_images_list("outputs/visualization/128x128")

X = torch.cat(images, dim=0).to(device)
y = torch.cat(labels, dim=0).to(device)

model_name = "resnext"
model = initialize_model(model_name, 15)

FILE = f"outputs/models/{model_name}/model.pth"
checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])

show_saliency_maps(X, y, classes, model, model_name)
