import os
import pickle
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


META = {
    'CIFAR10' : (CIFAR10, cifar_transform, 32, 3, 10),
    'MNIST' : (MNIST, mnist_transform, 28, 1, 10)
}

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, name, batch_size, num_workers, dataset_root=None) -> None:
        super(ImageDataModule).__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        root = dataset_root if dataset_root else '/'

        self.source = train_dir
        self.sink = os.path.join(root, name)
        self.name = name
        self.height = META[name][2]
        self.channels = META[name][3]
        self.classes = META[name][4]

        self.batch = batch_size
        self.workers = num_workers

    def prepare_data(self):
        pass 

    def setup(self, stage: Optional[str] = None): # Will make this cleaner later
        if stage == "fit" or stage is None:
            with open(self.source, 'rb') as f:
                x, y, _ = pickle.load(f)
                if self.name == 'MNIST':
                    x = np.expand_dims(x, axis=1)
                else:
                    x = np.einsum('ijkl->iljk', x)
                print(x.shape)
                ds, transform, _, _, _ = META[self.name]
            self.train, self.validate = TensorDataset(torch.Tensor(x), torch.Tensor(y)), ds(self.sink, train=False, download=True, transform=transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            ds, transform, _, _, _ = META[self.name]

            self.test = ds(self.sink, train=False, download=True, transform=transform)
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)