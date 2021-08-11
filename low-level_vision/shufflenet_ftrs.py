import numpy as np
from utils.knn import KNearestNeighbor
from tqdm import tqdm
import os
from sklearn.decomposition import PCA

root = "/home/serfani/Documents/atex/outputs"
X_train = np.loadtxt(os.path.join(
    root, 'train_shufflenet_ftrs.txt'), dtype=np.float64, delimiter=',')
y_train = np.loadtxt(os.path.join(
    root, 'train_shufflenet_lbls.txt'), delimiter=',').astype(int)

X_val = np.loadtxt(os.path.join(
    root, 'val_shufflenet_ftrs.txt'), dtype=np.float64, delimiter=',')
y_val = np.loadtxt(os.path.join(
    root, 'val_shufflenet_lbls.txt'), delimiter=',').astype(int)

print(X_train.shape, X_val.shape)
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
