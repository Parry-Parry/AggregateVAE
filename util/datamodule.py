import os
from typing import Optional

import pytorch_lightning as pl
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, emnist_normalization
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms 

import numpy as np

mnist_transform=transforms.Compose([transforms.ToTensor(),
                            emnist_normalization('mnist')
                            ])
cifar_transform=transforms.Compose([transforms.ToTensor(),
                            cifar10_normalization()
                            ])


NAMES = {
    'CIFAR10' : (CIFAR10, cifar_transform),
    'MNIST' : (MNIST, mnist_transform)
}

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, train, name, batch_size, num_workers, dataset_root=None) -> None:
        super(ImageDataModule).__init__()

        root = dataset_root if dataset_root else '/'

        x, y = train

        self.train_ds = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        ds, transform = NAMES[name]
        self.test_ds = ds(os.path.join(root, name), train=False, transform=transform)

        self.batch = batch_size
        self.workers = num_workers

    def prepare_data(self):
        pass 

    def setup(self, stage: Optional[str] = None): # Will make this cleaner later
        if stage == "fit" or stage is None:
            self.train, self.validate = self.train_ds, self.test_ds

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.test_ds
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)