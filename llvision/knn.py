import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io

from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv
from skimage.measure import block_reduce

# from skimage.util import img_as_float
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from classifiers import KNearestNeighbor
from tsne import tsne

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'


############################# ANALYSIS #############################
# # AdaBoost

from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

base_estimator = DecisionTreeClassifier(
    criterion='entropy', max_depth=1)  # criterion="gini"
# base_estimator = RandomForestClassifier(max_depth=1, random_state=0)

clf = AdaBoostClassifier(base_estimator=base_estimator,
                         n_estimators=100, random_state=0)
clf.fit(X_train, y_train)


y_val_pred = clf.predict(X_val)

print("-----------------------------------------------Before using AdaBoost------------------------------------------")
print(clf.score(X_val, y_val))  # prediction accuracy rate
# Contains the accuracy rate, recall rate and other information tables
print(metrics.classification_report(y_val, y_val_pred))
print(metrics.confusion_matrix(y_val, y_val_pred))  # Confusion matrix


feats = clf.decision_function(X_val)

# print(feats)
a = clf.feature_importances_
a_ = np.sort(a)[::-1]

print(a_[:10])
exit()


def tsne_plot(Y, labels, classes=list()):
    NUM_COLORS = len(classes)
    cm = plt.get_cmap('gist_rainbow')
    cidx = 0
    fig, ax = plt.subplots()
    markers = ["o", "x", "*", "+", 'd', "o", "x",
               "*", "+", 'd', "o", "x", "*", "+", 'd']
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS)
                             for i in range(NUM_COLORS)])
    for idx_, class_ in enumerate(classes):
        idx = np.sum(idx_ == labels)
        cidx += idx
        iidx = cidx - idx
        # print(iidx, cidx)
        ax.scatter(Y[iidx: cidx, 0],
                   Y[iidx:cidx:, 1], label=class_, marker=markers[idx_])
    ax.legend()
    ax.grid(True)

    plt.show()


# path = "./tsne3D_hsv_val_1000.txt"
# data = np.loadtxt(path, delimiter=',')
# print(data.shape)

since = time.time()
Y = tsne(X_train, 3, 50, 20.0)
np.savetxt('./models/tsne/tsne3D_train_1000.txt', Y, delimiter=',')
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


def tsne_3dplot(Y, labels, classes=list()):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import colors

    NUM_COLORS = len(classes)
    cm = plt.get_cmap('gist_rainbow')
    cidx = 0

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    markers = ["o", "x", "*", "+", 'd', "o", "x",
               "*", "+", 'd', "o", "x", "*", "+", 'd']
    axis.set_prop_cycle(color=[cm(1. * i / NUM_COLORS)
                               for i in range(NUM_COLORS)])
    for idx_, class_ in enumerate(classes):
        idx = np.sum(idx_ == labels)
        cidx += idx
        iidx = cidx - idx
        # print(iidx, cidx)
        axis.scatter(Y[iidx: cidx, 0],
                     Y[iidx:cidx:, 1], Y[iidx:cidx:, 2], label=class_, marker=markers[idx_])

    axis.set_xlabel(r"$1^{st}$ dim")
    axis.set_ylabel(r"$2^{nd}$ dim")
    axis.set_zlabel(r"$3^{rd}$ dim")
    axis.legend()
    axis.grid(True)

    plt.show()


# tsne_3dplot(Y, y_train, atex["classes"])
# Y = np.loadtxt("./models/tsne/tsne_hsv_train_1000.txt", delimiter=',')
# tsne_plot(Y, y_val, atex["classes"])
