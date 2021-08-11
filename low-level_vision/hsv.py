import numpy as np
from dataloader import dataloader
from utils.knn import KNearestNeighbor
from tqdm import tqdm

atex = dataloader(as_gray=False, norm=False, hsv=True)

X_train = atex["train"]["data"]
y_train = atex["train"]["target"]
X_val = atex["val"]["data"]
y_val = atex["val"]["target"]


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))


classifier = KNearestNeighbor()
k_choices = [1, 3, 5, 8, 15, 50, 70, 100, 200, 300, 500]

k_to_acc = {}
for k in tqdm(k_choices, desc='KNN Progress'):

    # use of k-nearest-neighbor algorithm
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_val, k=k, method="l2n")

    # Compute the fraction of correctly predicted examples
    num_correct = np.sum(y_pred == y_val)
    accuracy = float(num_correct) / X_val.shape[0]
    k_to_acc[k] = accuracy

# Print the computed accuracies
for k, acc in k_to_acc.items():
    print('k: %d \t accuracy: %f' % (k, acc))
