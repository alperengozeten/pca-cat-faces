import os
import numpy as np

from PIL import Image
from os import path
from numpy.linalg import eig
import matplotlib.pyplot as plt

# get the current working directory
ROOT_DIR = path.abspath(os.curdir)
DATA_DIR = path.join(ROOT_DIR, 'afhq_cat')

def load_single_image(image_path : str) -> np.ndarray:
    resizedImage = Image.open(image_path).resize((64,64), Image.BILINEAR) # 
    resizedImage = np.asarray(resizedImage, dtype=np.float64) # change from uint8 to float64
    resizedImage = np.resize(resizedImage, (4096, 3)) # flatten the first two dimensions
    return resizedImage

def load_all_images(dataset_path: str) -> np.ndarray:
    # stack the image data
    return np.asarray([load_single_image(path.join(DATA_DIR, os.fsdecode(file))) for file in os.listdir(dataset_path)]) 

load_single_image(path.join(DATA_DIR, 'flickr_cat_000002.jpg'))
out = load_all_images(DATA_DIR)
print('Dataset has been loaded succesfully!')

class PCA:
    def __init__(self) -> None:
        self.mean = None

    def apply(self, X: np.ndarray):
        X = np.asarray(X).copy()
        self.mean = np.mean(X, axis=0, keepdims=True)
        
        # normalize the data
        X = X - self.mean

        #calculate the covariance matrix
        covX = (X.T @ X) / X.shape[0]
        self.cov = covX

        # calculate the eigenvalues and eigenvectors
        eigenVals, eigenVectors = eig(covX)

        # sort the eigenvalues in descending order
        idx = eigenVals.argsort()[::-1]   
        eigenVals = eigenVals[idx]
        eigenVectors = eigenVectors[:,idx]

        # store the eigenvalues
        self.sorted_eigVals = eigenVals
        self.sorted_eigVectors = eigenVectors
    
    def find_explained_variance(self, k : int):
        return self.sorted_eigVals[0:10] / np.sum(self.sorted_eigVals)
    
    # returns the index where the cumulative explained variance exceeds f, which is between 0 and 1
    def cumulative_explained_variance(self, f : float):
        if f < 0 or f > 1:
            print('f is out of the interval [0, 1]')
            return -1

        # get the cumulative ratios
        cum_ratios = np.cumsum(self.sorted_eigVals) / np.sum(self.sorted_eigVals)
        return (np.searchsorted(cum_ratios, f) + 1) # +1 since this function returns the index
    
    # get the first k PCs after scaling them with min-max
    def min_max_scaled_components(self, k : int):
        result_vectors = []
        for i in range(k):
            vec = self.sorted_eigVectors[:, i]
            vec = np.reshape(vec, (64, 64))
            vec = (vec - vec.min()) / (vec.max() - vec.min())
            result_vectors.append(vec)
        return np.stack(result_vectors, axis=0)
    
    def check(self):
        return (self.cov @ self.sorted_eigVectors[:, 0]) - self.sorted_eigVals[0] * self.sorted_eigVectors[:, 0]
    
    def transform_image(self, X : np.ndarray, k: int):
        components = self.sorted_eigVectors[:, 0: k]
        return (X - self.mean).reshape((1, 4096)) @ components
    
    def reconstruct_image(self, X : np.ndarray, k: int):
        components = self.sorted_eigVectors[:, 0: k]
        return X @ components.T + self.mean

r_pca = PCA()
r_pca.apply(out[:, :, 0])

g_pca = PCA()
g_pca.apply(out[:, :, 1])

b_pca = PCA()
b_pca.apply(out[:, :, 2])

# Question 1.1
print('------- Red Channel -------\nIndividual explained variances of first 10 PCs:')
r_explained_first = r_pca.find_explained_variance(10)
print(r_explained_first)
print('Total explained variance by first 10 components: ' + str(np.sum(r_explained_first)))
print('The minimum number of principal components to ensure %70 explained variance: ' + str(r_pca.cumulative_explained_variance(0.7)))

print('------- Green Channel -------\nIndividual explained variances of first 10 PCs:')
g_explained_first = g_pca.find_explained_variance(10)
print(g_explained_first)
print('Total explained variance by first 10 components: ' + str(np.sum(g_explained_first)))
print('The minimum number of principal components to ensure %70 explained variance: ' + str(g_pca.cumulative_explained_variance(0.7)))

print('------- Blue Channel -------\nIndividual explained variances of first 10 PCs:')
b_explained_first = b_pca.find_explained_variance(10)
print(b_explained_first)
print('Total explained variance by first 10 components: ' + str(np.sum(b_explained_first)))
print('The minimum number of principal components to ensure %70 explained variance: ' + str(b_pca.cumulative_explained_variance(0.7)))

# Question 1.2
r_components = r_pca.min_max_scaled_components(10)
g_components = g_pca.min_max_scaled_components(10)
b_components = b_pca.min_max_scaled_components(10)

stacked_components = np.stack([r_components, g_components, b_components], axis=-1)

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8, 20))
for index in range(len(stacked_components)):
    row = index // 2
    col = index % 2
    ax[row, col].set_axis_off()
    component = stacked_components[index, ...]
    component = component.reshape((64, 64, -1))
    ax[row, col].imshow(component)
fig.tight_layout()
plt.suptitle('The First 10 Principal Components')
plt.show()

# Question 1.3
# Reconstruct the second image by using different number of principal components
orig_image = load_single_image(path.join(DATA_DIR, 'flickr_cat_000003.jpg'))

def plot_image(im: np.ndarray, title):
    plt.figure()
    im = im.reshape((64, 64, -1))

    # normalize each channel of the image
    channel1 = im[:, :, 0].reshape((64, 64))
    channel1 = (channel1 - channel1.min()) / (channel1.max() - channel1.min())

    channel2 = im[:, :, 1].reshape((64, 64))
    channel2 = (channel2 - channel2.min()) / (channel2.max() - channel2.min())

    channel3 = im[:, :, 2].reshape((64, 64))
    channel3 = (channel3 - channel3.min()) / (channel3.max() - channel3.min())

    im = np.stack([channel1, channel2, channel3], axis=-1)
    #im = (im - im.min()) / (im.max() - im.min())
    plt.axis('off')
    plt.title(title)
    plt.imshow(im)
    plt.show()

plot_image(orig_image, title='Original Image')

# Set of k values
k_list = [1, 50, 250, 500, 1000, 4096]

for k in k_list:
    r_transformed = r_pca.transform_image(orig_image[:, 0], k)
    r_reconstructed = r_pca.reconstruct_image(r_transformed, k)

    g_transformed = g_pca.transform_image(orig_image[:, 1], k)
    g_reconstructed = g_pca.reconstruct_image(g_transformed, k)

    b_transformed = b_pca.transform_image(orig_image[:, 2], k)
    b_reconstructed = b_pca.reconstruct_image(b_transformed, k)

    reconstructed_image = np.stack([r_reconstructed, g_reconstructed, b_reconstructed], axis=-1)

    plot_image(reconstructed_image, title=f'Reconstruction With k = {k} PCs')

    # Check if original and reconstructed images are identical within a tolerance
    if k == 4096:
        print(np.allclose(reconstructed_image, orig_image))
       