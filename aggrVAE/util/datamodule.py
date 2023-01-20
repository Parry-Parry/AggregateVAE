import os
import pickle
from typing import Optional
from PIL import Image

import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms 

import numpy as np

def apply_transforms_tensor(x, t):
    x = (x * 255).astype(np.uint8)

    if len(x.shape) == 3: mode = 'L'
    else: mode = 'RGB'
    
    tensors = []
    for i in range(x.shape[0]):
        tensors.append(t(Image.fromarray(x[i], mode)))
    
    return torch.stack(tensors)

class AggrMNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, batch_size, num_workers, dataset_root=None) -> None:
        super(AggrMNISTDataModule).__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        root = dataset_root if dataset_root else '/'
        self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                            transforms.Normalize(mean=0.173, std=0.332)
                            ])
        self.source = train_dir
        self.sink = os.path.join(root, 'MNIST')
        self.name = 'MNIST'
        self.height = 224
        self.channels = 1
        self.classes = 10

        self.batch = batch_size
        self.workers = num_workers

    def prepare_data(self):
        MNIST(self.sink, train=False, download=True, transform=self.transform)

    def setup(self, stage: Optional[str] = None): # Will make this cleaner later
        if stage == "fit" or stage is None:
            with open(self.source, 'rb') as f:
                x, y, _ = pickle.load(f)
            x = apply_transforms_tensor(x, self.transform)

            _, val = torch.utils.data.random_split(MNIST(self.sink, train=False, download=True, transform=self.transform), [9500, 500])

            self.train, self.validate = TensorDataset(x, torch.Tensor(y)), val

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = MNIST(self.sink, train=False, download=True, transform=self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)

class AggrCIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, train_dir, batch_size, num_workers, dataset_root=None) -> None:
        super(AggrCIFAR10DataModule).__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        root = dataset_root if dataset_root else '/'
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                            ])

        self.source = train_dir
        self.sink = os.path.join(root, 'CIFAR10')
        self.name = 'CIFAR10'
        self.height = 224
        self.channels = 3
        self.classes = 10

        self.batch = batch_size
        self.workers = num_workers

    def prepare_data(self):
        CIFAR10(self.sink, train=False, download=True, transform=self.transform)

    def setup(self, stage: Optional[str] = None): # Will make this cleaner later
        if stage == "fit" or stage is None:
            with open(self.source, 'rb') as f:
                x, y, _ = pickle.load(f)
            x = np.einsum('ijkl->iljk', x)
            x = apply_transforms_tensor(x, self.transform)
            _, val = torch.utils.data.random_split(CIFAR10(self.sink, train=False, download=True, transform=self.transform), [9500, 500])

            self.train, self.validate = TensorDataset(x, torch.Tensor(y)), val

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CIFAR10(self.sink, train=False, download=True, transform=self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)