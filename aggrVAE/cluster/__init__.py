import faiss
import torch
import numpy as np

class Kmeans:
    def __init__(self, 
                 spherical : bool = True, 
                 verbose : bool = False, 
                 seed : int = 42, 
                 gpu : bool = False) -> None:
        self.spherical = spherical
        self.verbose = verbose
        self.seed = seed
        self.gpu = False if not gpu else torch.cuda.device_count()

        self.centroids = None
        self.index = None

    def fit(self, X : np.ndarray, K : int = 50, iters : int = 20) -> None:
        kmeans = faiss.Kmeans(X.shape[1], 
                              K, 
                              niter=iters, 
                              spherical=self.spherical, 
                              verbose=self.verbose, 
                              seed=self.seed, 
                              gpu=self.gpu, 
                              min_points_per_centroid=X.shape[0] // K)
        kmeans.train(X)
        self.centroids = kmeans.centroids
        self.index = kmeans.index

        return None

    def fit_transform(self, X : np.ndarray, K : int = 50, iters : int = 20) -> np.ndarray:
        self.fit(X, K, iters)
        _, I = self.index.search(X, 1)
        return I.reshape(-1)