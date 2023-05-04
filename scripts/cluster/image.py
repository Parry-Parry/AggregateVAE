from fire import Fire
import numpy as np
from aggrVAE.util import load_img
from aggrVAE.cluster import Kmeans

def main(dataset : str, 
         datastore : str, 
         outdir : str, 
         K : int = 50, 
         n_class : int = 10, 
         verbose : bool = False, 
         seed : int = 42, 
         gpu : bool = False):
    train, _ = load_img(dataset, datastore)
    X = train.data.numpy()
    X = X.reshape(X.shape[0], -1)
    y = train.targets.numpy()

    per_class = K // n_class

    new_x = np.zeros(K, X.shape[-1])
    new_y = np.zeros(K, dtype=np.int64)

    kmeans = Kmeans(verbose=verbose, seed=seed, gpu=gpu)

    for i in range(np.max(y) + 1):
        idx = np.where(y == i)[0]
        X_i = X[idx]
        kmeans.fit(X_i, per_class)
        centroids = kmeans.centroids
        new_x[i * per_class : (i + 1) * per_class] = centroids
        new_y[i * per_class : (i + 1) * per_class] = i

    np.savez(outdir, X=new_x, y=new_y)

if __name__ == '__main__':
    Fire(main)