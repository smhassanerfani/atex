import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # covariance
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[:self.n_components]

        # store the frist n eigenvectors

    def transform(self, X):
        X -= self.mean
        return np.matmul(X, self.components.T)