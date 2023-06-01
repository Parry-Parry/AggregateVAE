import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms 

from .util import sparse_convert, apply_transforms_tensor


import numpy as np
import pandas as pd

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

    def setup(self, stage: Optional[str] = None): 
        import logging
        if stage == "fit" or stage is None:
            with np.load(self.source, allow_pickle=True) as data:
                x = apply_transforms_tensor(np.reshape(data['X'], (-1, 28, 28)), self.transform)
                y = torch.Tensor(data['y'])

            test, val = random_split(MNIST(self.sink, train=False, download=True, transform=self.transform), [8000, 2000])
    
            self.train, self.test, self.validate = TensorDataset(x.view(-1, self.channels, self.height, self.height), y), test, val
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)
    
class ReconsMNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, batch_size, num_workers, dataset_root=None, epsilon=0.01, p=5) -> None:
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

        self.epsilon = epsilon
        self.p = p

    def prepare_data(self):
        MNIST(self.sink, train=False, download=True, transform=self.transform)

    def setup(self, stage: Optional[str] = None): 
        if stage == "fit" or stage is None:
            data = np.load(self.source, allow_pickle=True)
            x = apply_transforms_tensor(np.reshape(data['X'], (-1, 28, 28)), self.transform)
            x = torch.tile(x, (self.p, 1, 1))
            y = torch.tile(torch.Tensor(data['y']), (self.p, ))
            x = x + torch.Tensor(x.shape).uniform_(-self.epsilon, self.epsilon)

            test, val = torch.utils.data.random_split(MNIST(self.sink, train=False, download=True, transform=self.transform), [8000, 2000])
    
            self.train, self.test, self.validate = TensorDataset(x.view(-1, self.channels, self.height, self.height), y), test, val
            
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

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            data = np.load(self.source, allow_pickle=True)
            x = apply_transforms_tensor(np.reshape(data['X'], (-1, 32, 32, 3)), self.transform)

            test, val = random_split(CIFAR10(self.sink, train=False, download=True, transform=self.transform), [8000, 2000])
            
            self.train, self.test, self.validate = TensorDataset(x.view(-1, self.channels, self.height, self.height), torch.Tensor(data['y'])), test, val
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)

class ReconsCIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, train_dir, batch_size, num_workers, dataset_root=None, epsilon=0.01, p=5) -> None:
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

        self.epsilon = epsilon
        self.p = p

    def prepare_data(self):
        CIFAR10(self.sink, train=False, download=True, transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            data = np.load(self.source, allow_pickle=True)
            x = apply_transforms_tensor(np.reshape(data['X'], (-1, 32, 32, 3)), self.transform)
            x = torch.tile(x, (self.p, 1, 1, 1))
            y = torch.tile(torch.Tensor(data['y']), (self.p, ))

            x = x + torch.Tensor(x.shape).uniform_(-self.epsilon, self.epsilon)
            test, val = random_split(CIFAR10(self.sink, train=False, download=True, transform=self.transform), [8000, 2000])

            self.train, self.test, self.validate = TensorDataset(x.view(-1, self.channels, self.height, self.height), y), test, val
            
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, dataset_root=None):
        super().__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        root = dataset_root if dataset_root else '/'
        self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                            transforms.Normalize(mean=0.173, std=0.332)
                            ])
        self.sink = os.path.join(root, 'MNIST')
        self.name = 'MNIST'
        self.height = 224
        self.channels = 1
        self.classes = 10

        self.batch = batch_size
        self.workers = num_workers
    
    def prepare_data(self):
        MNIST(self.sink, train=False, download=True, transform=self.transform)
        MNIST(self.sink, train=True, download=True, transform=self.transform)

    def setup(self, stage: Optional[str] = None): 
        train = MNIST(self.sink, train=True, download=True, transform=self.transform)
        test, validate = random_split(MNIST(self.sink, train=False, download=True, transform=self.transform), [8000, 2000])

        self.train, self.validate, self.test = train, validate, test
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, dataset_root=None):
        super().__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        root = dataset_root if dataset_root else '/'
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                            ])
        self.sink = os.path.join(root, 'CIFAR10')
        self.name = 'CIFAR10'
        self.height = 224
        self.channels = 3
        self.classes = 10

        self.batch = batch_size
        self.workers = num_workers
    
    def prepare_data(self):
        CIFAR10(self.sink, train=False, download=True, transform=self.transform)
        CIFAR10(self.sink, train=True, download=True, transform=self.transform)

    def setup(self, stage: Optional[str] = None): 
        train = CIFAR10(self.sink, train=True, download=True, transform=self.transform)
        test, validate = random_split(CIFAR10(self.sink, train=False, download=True, transform=self.transform), [8000, 2000])

        self.train, self.validate, self.test = train, validate, test
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)

class TabularDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, dataset_root=None):
        super().__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        root = dataset_root if dataset_root else '/'

        self.train = None 
        self.test = None
    
        self.source = root
        self.features = None
        self.classes = 2

        self.batch = batch_size
        self.workers = num_workers

    def setup(self, stage: Optional[str] = None): 
        train = pd.read_csv(os.path.join(self.source, f'train.csv'), index_col=0)
        test = pd.read_csv(os.path.join(self.source, 'test.csv'), index_col=0)

        self.features = len(train.columns) - 1 # Remove target

        x, y = sparse_convert(train, 'target')
        train = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        x, y = sparse_convert(test, 'target')
        tmp = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        test, validate = random_split(tmp, [0.8, 0.2])

        self.train, self.validate, self.test = train, validate, test

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)
    
class AggrTabularDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, dataset_root=None):
        super().__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        root = dataset_root if dataset_root else '/'

        self.train = None 
        self.test = None
    
        self.source = root
        self.features = None
        self.classes = None

        self.batch = batch_size
        self.workers = num_workers

    def setup(self, stage: Optional[str] = None): 
        if stage == "fit" or stage is None:
            data = np.load(self.source, allow_pickle=True)
            x, y = data['x'], data['y']
        train = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        self.features = data['x'].shape[-1]
        self.classes = data['y'].shape[-1]

        test = pd.read_csv(os.path.join(self.source, 'test.csv'), index_col=0)
        x, y = sparse_convert(test, 'target')
        tmp = TensorDataset(torch.Tensor(x), torch.Tensor(y))

        test, validate = random_split(tmp, [0.8, 0.2])

        self.train, self.validate, self.test = train, validate, test

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)

class ReconsTabularDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, dataset_root=None, epsilon=0.01, p=5):
        super().__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        root = dataset_root if dataset_root else '/'

        self.train = None 
        self.test = None
    
        self.source = root
        self.features = None
        self.classes = None

        self.batch = batch_size
        self.workers = num_workers

        self.epsilon = epsilon
        self.p = p

    def setup(self, stage: Optional[str] = None): 
        train = pd.read_csv(os.path.join(self.source, f'train.csv'), index_col=0)
        self.features = len(train.columns) - 1 # Remove target

        x, y = sparse_convert(train, 'target')

        x = torch.tile(torch.Tensor(x), (self.p, 1))
        y = torch.tile(torch.Tensor(x), (self.p, 1))

        train = TensorDataset(x, y)

        test = pd.read_csv(os.path.join(self.source, 'test.csv'), index_col=0)
        x, y = sparse_convert(test, 'target')
        tmp = TensorDataset(torch.Tensor(x), torch.Tensor(y))

        test, validate = random_split(tmp, [0.8, 0.2])
        self.train, self.validate, self.test = train, validate, test

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch, num_workers=self.workers)