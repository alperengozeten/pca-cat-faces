import os
import numpy as np

from PIL import Image
from os import path
from typing import Iterator
from numpy.linalg import eig

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
print(type(out))
print(out.shape)
print(out[0].shape)

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

        # calculate the eigenvalues and eigenvectors
        w, v = eig(covX)

        # sort the eigenvalues in descending order
        w_sorted = np.sort(w)[::-1]
        self.sorted_eigVals = w_sorted
    
    def find_explained_variance(self, k : int):
        return self.sorted_eigVals[0:10] / np.sum(self.sorted_eigVals)
    
    # returns the index where the cumulative explained variance exceeds f, which is between 0 and 1
    def cumulative_explained_variance(self, f : float):
        if f < 0 or f > 1:
            print('f is out of the interval [0, 1]')
            return -1

        # get the cumulative ratios
        cum_ratios = np.cumsum(self.sorted_eigVals) / np.sum(self.sorted_eigVals)
        return np.searchsorted(cum_ratios, f)

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
print('The total number of principal components to ensure %70 explained variance: ' + str(r_pca.cumulative_explained_variance(0.7)))

print('------- Green Channel -------\nIndividual explained variances of first 10 PCs:')
g_explained_first = g_pca.find_explained_variance(10)
print(g_explained_first)
print('Total explained variance by first 10 components: ' + str(np.sum(g_explained_first)))
print('The total number of principal components to ensure %70 explained variance: ' + str(g_pca.cumulative_explained_variance(0.7)))

print('------- Blue Channel -------\nIndividual explained variances of first 10 PCs:')
b_explained_first = b_pca.find_explained_variance(10)
print(b_explained_first)
print('Total explained variance by first 10 components: ' + str(np.sum(b_explained_first)))
print('The total number of principal components to ensure %70 explained variance: ' + str(b_pca.cumulative_explained_variance(0.7)))

# Question 1.2
