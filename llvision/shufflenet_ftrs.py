import numpy as np
from utils.knn import KNearestNeighbor
from tqdm import tqdm
import os


root = "/home/serfani/Documents/atex/outputs"
X_train = np.loadtxt(os.path.join(
    root, 'train_shufflenet_ftrs.txt'), delimiter=',')
y_train = np.loadtxt(os.path.join(
    root, 'train_shufflenet_lbls.txt'), delimiter=',').astype(int)

X_val = np.loadtxt(os.path.join(
    root, 'val_shufflenet_ftrs.txt'), delimiter=',')
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


# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# pca = PCA(n_components=100, random_state=88)
# pca.fit(X_train)
# x = pca.transform(X_train)


# # KMeans to evaluate the inetria  # change the metric!
# inertia_list = []
# for k in range(20):
#     kmn = KMeans(n_clusters=k, random_state=88)
#     kmn.fit(x)
#     pred = kmn.labels_
#     inertia_list.append(kmn.inertia_)

# fig, axes = plt.subplots(nrows=1, ncols=1)
# axes.plot(np.arange(2, 16), inertia_list, 'o--')
# plt.grid(True)
# plt.show()
