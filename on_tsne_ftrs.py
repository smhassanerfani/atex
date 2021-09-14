import numpy as np
from dataloader import ATeX
from sklearn.manifold import TSNE
from models.tsne import extract_sequence
from utils.visualize import savegif, plot_2d
# from utils.pca import PCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

font = {'font.family': 'Times New Roman', 'font.size': 10}
plt.rcParams.update(**font)

dataset = ATeX()

ftrs = np.loadtxt('./outputs/train_shufflenet_ftrs.txt', delimiter=',')
lbls = np.loadtxt('./outputs/train_shufflenet_lbls.txt', delimiter=',')

ftrs2 = np.loadtxt('./outputs/train_tsne_ftrs.txt', delimiter=',')

pca = PCA(n_components=20)
pca.fit(ftrs)
ftrs20 = pca.transform(ftrs)

for idx, name in enumerate(dataset.classes):
    print(idx, name)

    X1 = ftrs2[lbls == idx]
    X2 = ftrs20[lbls == idx]
    y = lbls[lbls == idx]

    range_n_clusters = [2, 3, 4]

    for n_clusters in range_n_clusters:

        clusterer = KMeans(n_clusters=n_clusters, random_state=15).fit(X1)
        cluster_labels = clusterer.labels_

        fig, axes = plt.subplots(n_clusters, 1)
        for lbl_, ax in enumerate(axes):
            if lbl_ != n_clusters - 1:
                ax.axes.xaxis.set_visible(False)
            sns.heatmap(X2[cluster_labels == lbl_],
                        cmap="YlGnBu", cbar=False, ax=ax)

        plt.savefig(
            f"./outputs/kmeans_results/heatmaps/{dataset.classes[idx]}_{n_clusters}", format="svg")

exit()
