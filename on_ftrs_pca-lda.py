import numpy as np
from dataloader import ATeX
from utils.pca import PCA
# from utils.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.visualize import plot_2d

dataset = ATeX()
features = np.loadtxt('./outputs/train_shufflenet_ftrs.txt', delimiter=',')
labels = np.loadtxt('./outputs/train_shufflenet_lbls.txt', delimiter=',')


# lda = LinearDiscriminantAnalysis(n_components=2)
# X_r = lda.fit(features, labels).transform(features)
# plot_2d(X_r, labels, dataset.classes, legend=True)

pca = PCA(2)
pca.fit(features)
X_r = pca.transform(features)
plot_2d(X_r, labels, dataset.classes)
