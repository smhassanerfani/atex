import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

# font = {'font.family': 'Times New Roman', 'font.size': 10}
# plt.rcParams.update(**font)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """

    model.to(device)

    # Make sure the model is in "test" mode
    model.eval()
    model.zero_grad()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # Perform a forward and backward pass through the model to compute the       #
    # gradient of the correct class score with respect to each input image. You  #
    # first want to compute the loss over the correct scores (we'll combine      #
    # losses across a batch by summing), and then compute the gradients with a   #
    # backward pass.                                                             #
    ##############################################################################
    # forward pass
    scores = model(X)
    _, preds = torch.max(scores, 1)
    scores = (scores.gather(1, y.view(-1, 1)).squeeze())

    # backward pass
    v = torch.FloatTensor([1.0] * scores.shape[0]).to(device)
    scores.backward(v)

    # saliency
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency, preds

def show_saliency_maps(X, y, classes, model, model_name):

    # Compute saliency maps for images in X
    saliency, preds = compute_saliency_maps(X, y, model)
    X = [deprocess(x) for x in X]
    X = [x.detach().cpu().numpy().transpose(1, 2, 0) for x in X]
    y = y.detach().cpu().numpy()
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().cpu().numpy()
    preds = preds.tolist()
    N = len(X)

    fig, axes = plt.subplots(nrows=2, ncols=N)
    fig.set_size_inches(12, 5)
    for i in range(N):
        axes[0, i].imshow(X[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(classes[y[i]])

        axes[1, i].imshow(saliency[i], cmap=plt.cm.hot)
        axes[1, i].axis('off')
        axes[1, i].set_title(classes[preds[i]])

    fig.suptitle(model_name)
    plt.show()
    # plt.savefig(f"./outputs/visualization/{model_name}.svg", dpi=150)
    # plt.close()


def get_images_list(path):
    imgs = list()
    lbls = list()
    clss = list()
    for root, dirs, files in os.walk(path, topdown=True):

        for idx, file in enumerate(files):
            if file.endswith(".jpg"):
                image = Image.open(os.path.join(root, file))
                image = preprocess(image)
                imgs.append(image)
                lbls.append(torch.tensor([idx]))
                clss.append(file.split(".")[0])

    return imgs, lbls, clss


def preprocess(img, size=128):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = T.Compose([
        # T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        # T.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled
