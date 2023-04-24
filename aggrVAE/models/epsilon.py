from abc import abstractmethod
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .classifier import GenericClassifier

gen_param = lambda x : nn.Parameter(torch.Tensor([x]))

class NeighbourhoodClassifier(GenericClassifier):
    def __init__(self, 
                 encoder, 
                 head,
                 epsilon=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = encoder
        self.head = head
    
    def noise(self, x):
        return x + torch.randn_like(x) * self.epsilon
    
    def forward(self, x, training=False):
        X = []
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        z = self.fc_z(x_encoded)
        y_hat = self.head(z)
        return y_hat

class EnsembleNeighbourhoodClassifier(GenericClassifier):
    agg = {
        'mean' : lambda x : torch.mean(x, dim=0),
        'max' : lambda x : torch.max(x, dim=0)[0], # fix this
    }
    def __init__(self, 
                 encoder,
                 head : callable,
                 num_heads : int = 2,
                 agg : str = 'mean',
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.head = nn.ModuleList([head(i) for i in range(num_heads)])
        self.num_heads = num_heads
        self.agg_func = self.agg[agg]
    
    def forward(self, x, training=False):
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        z = self.fc_z(x_encoded)
        inter_y = torch.stack([head(z) for head in self.head])
        y_hat = self.agg_func(inter_y)
        if training: return y_hat, inter_y
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        _, inter_y = self.forward(x, training=True)
        loss = torch.sum(torch.stack([self.loss_fn(_y, y) for _y in inter_y]))
        return {'loss' : loss}