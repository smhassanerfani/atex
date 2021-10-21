import numpy as np
from dataloader import dataloader
from skimage.feature import local_binary_pattern as lbp
from utils.knn import KNearestNeighbor
from tqdm import tqdm

rootdir="C:\\Users\\SERFANI\\Documents\\atex\\data\\atex"
atex = dataloader(as_gray=True, norm=False, hsv=False, rootdir=rootdir)

X_train = atex["train"]["data"]
y_train = atex["train"]["target"]
X_val = atex["val"]["data"]
y_val = atex["val"]["target"]


# Hyper-Parameters
METHOD = 'uniform'
radius = 1
n_points = 8 * radius


# lbp
X_train = map(lambda x: lbp(x, n_points, radius, METHOD), X_train)
X_train = np.asarray(list(X_train))

X_val = map(lambda x: lbp(x, n_points, radius, METHOD), X_val)
X_val = np.asarray(list(X_val))


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))


classifier = KNearestNeighbor()
k_choices = [1, 3, 5, 8, 15, 50, 70, 100, 200, 300, 500]

k_to_acc = {}
for k in tqdm(k_choices, desc='KNN Progress'):

    # use of k-nearest-neighbor algorithm
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_val, k=k, method="llv")

    # Compute the fraction of correctly predicted examples
    num_correct = np.sum(y_pred == y_val)
    accuracy = float(num_correct) / X_val.shape[0]
    k_to_acc[k] = accuracy

# Print the computed accuracies
for k, acc in k_to_acc.items():
    print('k: %d \t accuracy: %f' % (k, acc))
