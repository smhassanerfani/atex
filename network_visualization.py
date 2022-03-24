import torch
from utils.initialize_model import initialize_model
from utils.saliency import show_saliency_maps, get_images_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())


images, labels, classes = get_images_list("outputs/visualization/128x128")

X = torch.cat(images, dim=0).to(device)
y = torch.cat(labels, dim=0).to(device)

model_list = ["wide_resnet", "vgg", "squeezenet", "shufflenet", "resnext", "resnet", "mobilenet", "googlenet", "efficientnet-b7", "efficientnet-b0", "densenet"]

for model_name in model_list:

    model = initialize_model(model_name, num_classes=15, feature_extract=True, use_pretrained=True)

    FILE = f"outputs/models_v2/{model_name}/model.pth"
    checkpoint = torch.load(FILE)
    model.load_state_dict(checkpoint['model_state'])

    show_saliency_maps(X, y, classes, model, model_name)
