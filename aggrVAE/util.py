from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List
from .models.modules import Head 

@dataclass
class Log:
    epoch : int
    loss : dict
    val_metrics : dict

@dataclass
class LogStore:
    logs : List[Log]
    test_metrics : dict

def sparse_to_dense(df : pd.DataFrame, target : str, n_class):
    import torch
    labels = torch.Tensor((df.pop(target).values)).to(torch.int64)
    features = []
    for col in df.columns:
        tmp_array = np.array(df[col].values)
        features.append(tmp_array)
        #if col not in categorical: tmp_array = tmp_array / np.linalg.norm(tmp_array)
    x = np.stack(features, axis=1)
    return torch.Tensor(x), torch.nn.functional.one_hot(labels, num_classes=n_class)

def sparse_convert(df : pd.DataFrame, target : str, categorical : list = []):
    labels = np.array(df.pop(target).values)
    features = []
    for col in df.columns:
        tmp_array = np.array(df[col].values)
        if col not in categorical: tmp_array = tmp_array / np.linalg.norm(tmp_array)
        features.append(tmp_array)
    x = np.stack(features, axis=1)
    return x, labels

def apply_transforms_tensor(x, t):
    import torch
    from PIL import Image

    x = (x * 255).astype(np.uint8)
    if len(x.shape) == 3: mode = 'L'
    else: mode = 'RGB'
    
    tensors = [t(Image.fromarray(x[i], mode)) for i in range(x.shape[0])]
    
    return torch.stack(tensors)

def load_img(dataset : str, storage : str, download : bool = True):
    from torchvision.datasets import CIFAR10, MNIST
    from torchvision import transforms

    data = {
        'mnist': MNIST,
        'cifar10': CIFAR10
    }

    tform = {
        'mnist': transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                            transforms.Normalize(mean=0.173, std=0.332)
                            ]),
        'cifar10': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                            ])
    }

    train = data[dataset](storage, train=True, download=download, transform=tform[dataset])
    test = data[dataset](storage, train=False, download=download, transform=tform[dataset])

    return train, test

def load_tabular(path, header = False, cols = None, transform = None, target = 'label', categorical = []):
    from os.path import join

    if cols: 
        train = pd.read_csv(join(path, 'train.csv'), header=0 if header else None, usecols=cols)
        test = pd.read_csv(join(path, 'test.csv'), header=0 if header else None, usecols=cols)
    else: 
        train = pd.read_csv(join(path, 'train.csv'), header=0 if header else None)
        test = pd.read_csv(join(path, 'test.csv'), header=0 if header else None)

    if transform: 
        train = transform(train, target, categorical)
        test = transform(test, target, categorical)

    return train, test

def infer_labels(X : np.array, y : np.array, c : np.array, centroids : np.array):
    assert X.shape[0] == y.shape[0] == c.shape[0], 'X, y, c must have same length'
    labels = np.zeros(centroids.shape[0], dtype=np.int)
    for i in range(centroids.shape[0]):
        idx = np.where(c == i)
        if len(idx) == 0: continue
        cluster = y[idx]
        counts = np.bincount(cluster)
        labels[idx] = np.argmax(counts)
    return centroids, labels

def callable_head(in_dim : int, stack : List[int], n_class : int, **kwargs):
    def inner_func(i=0):
        return Head(in_dim, stack, n_class, i=i, **kwargs)
    
    return inner_func

def init_out(dir : str):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(os.path.join(dir, 'models')):
        os.makedirs(os.path.join(dir, 'models'))
    
def dump_logs(logs : LogStore, file):
    import json
    with open(file, 'w') as f:
        json.dump(logs, f, indent=4, default=lambda o: o.__dict__)