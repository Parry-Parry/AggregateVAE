from abc import abstractmethod
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

gen_param = lambda x : nn.Parameter(torch.Tensor([x]))

class genericClassifier(nn.Module):
    def __init__(self,
                 enc_dim : int = 200,
                 loss_fn : callable = F.cross_entropy,
                 latent_dim : int = 200,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.enc_dim = enc_dim
        self.loss_fn = loss_fn
        self.fc_z = nn.Linear(enc_dim, latent_dim)
    
    @abstractmethod
    def forward(self, x, training=False):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, training=True)
        loss = self.loss_fn(y_hat, y)
        return loss

class SequentialClassifier(genericClassifier):
    def __init__(self, 
                 encoder, 
                 head,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = encoder
        self.head = head
    
    def forward(self, x, training=False):
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        z = self.fc_z(x_encoded)
        y_hat = self.head(z)
        return y_hat

class EnsembleClassifier(genericClassifier):
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
        return loss