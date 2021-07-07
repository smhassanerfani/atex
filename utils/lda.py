import numpy as np
from numpy.core.fromnumeric import mean


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # S_W, S_B
        mean_total = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += np.matmul((X_c - mean_c).T, (X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_total).reshape(n_features, 1)
            S_B += n_c * np.matmul(mean_diff, mean_diff.T)

        A = np.matmul(np.linalg.inv(S_W), S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[:self.n_components]

    def transform(self, X):
        return np.matmul(X, self.linear_discriminants.T)
