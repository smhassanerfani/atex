import numpy as np
from dataloader import ATeX
from sklearn.manifold import TSNE
from models.tsne import extract_sequence
from utils.visualize import savegif

dataset = ATeX()
features = np.loadtxt('./outputs/train_shufflenet_ftrs.txt', delimiter=',')
labels = np.loadtxt('./outputs/train_shufflenet_lbls.txt', delimiter=',')

perplexity = 20
learning_rate = 200
n_iter = 300
exploration_n_iter = 100
method = "learning_rate"

tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate,
            n_iter=n_iter, verbose=2, method=method)
tsne._EXPLORATION_N_ITER = exploration_n_iter

Y_seq = extract_sequence(tsne, features)

lo = Y_seq.min(axis=0).min(axis=0).max()
hi = Y_seq.max(axis=0).max(axis=0).min()
limits = ([lo, hi], [lo, hi])

fig_name = "%s-%d-%d-tsne" % ("ATeX", 300, 100)
fig_path = "./outputs/%s.gif" % (fig_name)
savegif(Y_seq, labels, "t-SNE", fig_path, dataset.classes, limits=limits)
