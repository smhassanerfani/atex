import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from statsmodels.stats.weightstats import ztest

from dataloader import ATeX
from utils.visualize import boxplot
from torch.utils.data import DataLoader

# Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


font = {'font.family': 'Times New Roman', 'font.size': 14}
plt.rcParams.update(**font)

dataset = ATeX(as_gray=False)
atex = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

imgs = [np.array(atex.dataset[x][0]) for x in range(len(atex.dataset))]
imgs = np.array(imgs)

# Channel Selection
# imgs = imgs.transpose(0, 3, 1, 2)
# imgs = imgs[:, 2]
print(imgs.shape)

# DO NOT NEED THIS! USE `np.loadtxt`
# lbls = [np.array(atex.dataset[x][1]) for x in range(len(atex.dataset))]
# lbls = np.array(lbls)

rng = np.random.default_rng()
p = imgs.reshape(imgs.shape[0], -1)
ftrs50 = rng.choice(p, 50, replace=False, axis=1, shuffle=True)

# output = map(lambda x, y: ztest(x, x2=None, value=np.nanmean(y)), ftrs50, p)
# # output = [ztest, pval]
# output = np.array(list(output))
# print(np.nanmean(output, axis=0))

lbls = np.loadtxt('./outputs/train_shufflenet_lbls.txt', delimiter=',')
ftrs2 = np.loadtxt('./outputs/train_tsne_ftrs.txt', delimiter=',')

boxplot(ftrs50, lbls, dataset.classes)

for idx, name in enumerate(dataset.classes):
    if name == "glaciers":

        X1 = ftrs2[lbls == idx]
        X1 = StandardScaler().fit_transform(X1)

        X2 = ftrs50[lbls == idx]
        # X2 = StandardScaler().fit_transform(X2)

        y = lbls[lbls == idx]
        print(
            f"index number and total samples for \"{name}\": {idx}, {X1.shape[0]}")
        # #############################################################################
        # Compute DBSCAN

        print(20 * "-", "Compute DBSCAN", 20 * "-")

        db = DBSCAN(eps=0.3, min_samples=5).fit(X1)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        cluster_labels = db.labels_

        boxplot(X2, cluster_labels)

        n_clusters_ = len(set(cluster_labels)) - \
            (1 if -1 in cluster_labels else 0)
        n_noise_ = list(cluster_labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X1, cluster_labels))

        fig, (ax1, ax2) = plt.subplots(1, 2)

        fig.set_size_inches(12, 5)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X1) + (n_clusters_ + 1) * 10])

        silhouette_avg = metrics.silhouette_score(X1, cluster_labels)
        sample_silhouette_values = metrics.silhouette_samples(
            X1, cluster_labels)

        y_lower = 10
        for i in range(n_clusters_):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters_)

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters_)
        # RED used for noise.
        colors[cluster_labels == -1] = (1, 0, 0, 1)

        ax2.scatter(X1[:, 0], X1[:, 1], marker='o', s=50,
                    alpha=0.7, c=colors, linewidths=0.5, edgecolor='k')

        for i, c in enumerate(range(n_clusters_)):
            points_of_cluster = X1[cluster_labels == i, :]
            centers = np.mean(points_of_cluster, axis=0)

            ax2.scatter(centers[0], centers[1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            ax2.scatter(centers[0], centers[1], marker='$%d$' % i, alpha=1,
                        s=100, edgecolor='k')

        plt.title('Estimated number of clusters: %d' % n_clusters_)

        if n_clusters_ != 1:
            fig, axes = plt.subplots(n_clusters_, 1)
            fig.set_size_inches(6, 5)

            for lbl_, ax in enumerate(axes):

                if lbl_ != n_clusters_ - 1:
                    ax.xaxis.set_visible(False)

                sns.heatmap(X2[cluster_labels == (n_clusters_ - 1) - lbl_],
                            cmap="jet", cbar=False, ax=ax)

                ax.set_yticks(
                    [0, len(X2[cluster_labels == (n_clusters_ - 1) - lbl_])])
                ax.set_yticklabels(
                    (0, len(X2[cluster_labels == (n_clusters_ - 1) - lbl_])), rotation=0)
                ax.set_ylabel(f"{(n_clusters_ - 1) - lbl_}", rotation=0)

                print(
                    (n_clusters_ - 1) - lbl_, len(X2[cluster_labels == (n_clusters_ - 1) - lbl_]))

        # #############################################################################
        # Compute KMEANS
        print(20 * "-", "Compute KMEANS", 20 * "-")

        range_n_clusters = [n_clusters_]

        for n_clusters in range_n_clusters:

            print(f"Considered number of clusters: {n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, random_state=15).fit(X1)
            cluster_labels = kmeans.labels_
            fig, axes = plt.subplots(n_clusters, 1)
            fig.set_size_inches(6, 5)

            for lbl_, ax in enumerate(axes):

                if lbl_ != n_clusters - 1:
                    ax.axes.xaxis.set_visible(False)

                sns.heatmap(X2[cluster_labels == (n_clusters_ - 1) - lbl_],
                            cmap="jet", cbar=False, ax=ax)

                ax.axes.set_yticks(
                    [0, len(X2[cluster_labels == (n_clusters_ - 1) - lbl_])])
                ax.axes.set_yticklabels(
                    (0, len(X2[cluster_labels == (n_clusters_ - 1) - lbl_])))
                ax.axes.set_ylabel(f"{(n_clusters_ - 1) - lbl_}", rotation=0)

                print((n_clusters_ - 1) - lbl_,
                      len(X2[cluster_labels == (n_clusters_ - 1) - lbl_]))

        # plt.savefig(f"./outputs/kmeans_results/heatmaps/{dataset.classes[idx]}_{n_clusters}", format="svg")
        # boxplot(X2, cluster_labels)
        #############################################################################
        # Compute KMEANS
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(12, 5)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X1) + (n_clusters + 1) * 10])

        cluster_inertia = kmeans.inertia_

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = metrics.silhouette_score(X1, cluster_labels)
        print("For n_clusters: {0:1d}, The average silhouette_score: {1:.4f}, inetria: {2:.4f}".format(
            n_clusters, silhouette_avg, cluster_inertia))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = metrics.silhouette_samples(
            X1, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(
            cluster_labels.astype(float) / n_clusters)

        ax2.scatter(X1[:, 0], X1[:, 1], marker='o', s=50,
                    alpha=0.7, c=colors, linewidths=0.5, edgecolor=(0, 0, 0, 1))

        # Labeling the clusters
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=100, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

plt.show()
