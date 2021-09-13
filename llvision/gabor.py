import numpy as np
from dataloader import dataloader
from utils.knn import KNearestNeighbor
from sklearn.decomposition import PCA
from tqdm import tqdm

from skimage.filters import gabor_kernel
from skimage.measure import block_reduce
from utils.transforms import power
from utils.visualization import plot_samples
import concurrent.futures
import time

as_gray = True
norm = False

atex = dataloader(as_gray=as_gray, norm=norm, hsv=False)

# plot_samples(atex, num_of_samples=10)
# exit()


X_train = atex["train"]["data"]
y_train = atex["train"]["target"]
X_val = atex["val"]["data"]
y_val = atex["val"]["target"]

gjet_train = []
gjet_val = []
sigma = np.pi


def power_executer(x):
    return power(x, kernel, as_gray=as_gray)


def block_reduce_executor(x):
    return block_reduce(x, (2, 2), func=np.max)


new_xtrain = []

for mu in tqdm([0, 1, 2, 3, 4, 5, 6, 7], desc='mu'):
    for nu in [0, 1, 2, 3, 4]:

        theta = (mu / 8.) * np.pi
        frequency = (np.pi / 2) / (np.sqrt(2)) ** nu
        kernel = gabor_kernel(frequency, theta=theta,
                              sigma_x=sigma, sigma_y=sigma)
        since = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Mapping conv func of designed kernel on all images
            _X_train = executor.map(power_executer, X_train)
            # Mapping conv func of designed kernel on all images
            _X_train = executor.map(block_reduce_executor, _X_train)

            _X_val = executor.map(power_executer, X_val)
            _X_val = executor.map(block_reduce_executor, _X_val)

            _X_train = np.asarray(list(_X_train))
            gjet_train.append(_X_train)

            _X_val = np.asarray(list(_X_val))
            gjet_val.append(_X_val)

            time_elapsed = time.time() - since

        # # without multiprocessing
        # # Mapping conv func of designed kernel on all images
        # _X_train = map(lambda x: power(x, kernel, as_gray=as_gray), X_train)
        # # Downsampling the conv layer output
        # _X_train = map(lambda x: block_reduce(
        #     x, (2, 2), func=np.max), _X_train)

        # _X_train = np.asarray(list(_X_train))
        # gjet_train.append(_X_train)

        # _X_val = map(lambda x: power(x, kernel, as_gray=as_gray), X_val)
        # _X_val = map(lambda x: block_reduce(x, (2, 2), func=np.max), _X_val)

        # _X_val = np.asarray(list(_X_val))
        # gjet_val.append(_X_val)

        print(
            f"kernel size: {kernel.shape}, mu: {mu}, nu:{nu}, image size: {_X_train.shape}, process-time: {time_elapsed}")


gjet_train = np.array(gjet_train)
gjet_train = gjet_train.transpose(1, 2, 3, 0)

gjet_val = np.array(gjet_val)
gjet_val = gjet_val.transpose(1, 2, 3, 0)

X_train = gjet_train.reshape(gjet_train.shape[0], -1)
X_val = gjet_val.reshape(gjet_val.shape[0], -1)

print("X_train: ", X_train.shape)
print("X_val: ", X_val.shape)

pca = PCA(n_components=1024, random_state=88)
X_train = pca.fit_transform(X_train)
X_val = pca.fit_transform(X_val)

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
