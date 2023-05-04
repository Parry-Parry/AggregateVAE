from typing import List
from fire import Fire
import numpy as np
from ...cluster import Kmeans
from ...util import load_tabular, sparse_convert

def main(dataset : str, 
         outdir : str, 
         K : int = 50, 
         n_class : int = 10, 
         target : str = 'label', 
         categorical : List[str] = [], 
         verbose : bool = False, 
         seed : int = 42, 
         gpu : bool = False):
    train, _ = load_tabular(dataset, transform=sparse_convert, target=target, categorical=categorical)
    X, y = train
    kmeans = Kmeans(verbose=verbose, seed=seed, gpu=gpu)

    per_class = K // n_class

    new_x = np.zeros(K, X.shape[-1])
    new_y = np.zeros(K, dtype=np.int64)

    kmeans = Kmeans(verbose=verbose, seed=seed, gpu=gpu)

    for i in range(np.max(y) + 1):
        idx = np.where(y == i)
        X_i = X[idx]
        kmeans.fit(X_i, per_class)
        centroids = kmeans.centroids
        new_x[i * per_class : (i + 1) * per_class] = centroids
        new_y[i * per_class : (i + 1) * per_class] = i

    np.savez(outdir, X=new_x, y=new_y)



if __name__ == '__main__':
    Fire(main)