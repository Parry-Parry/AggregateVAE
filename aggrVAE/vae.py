from abc import abstractmethod
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

gen_param = lambda x : nn.Parameter(torch.Tensor([x]))

"""
genericVAE takes an encoder structure & classification head as arguments 
and intializes with gumbel softmax sampling in the latent distribution
"""
class genericVAE(nn.Module):
    def __init__(self, 
                 enc_dim : int = 200, 
                 latent_dim : int = 4, 
                 cat_dim : int = 10, 
                 t : float = 0.5, 
                 rate : float = 3e-5, 
                 min_t : float = 0.2,
                 kl_coeff : float = 1.,
                 interval = 100,
                 **kwargs):
        super(genericVAE, self).__init__(**kwargs)
        self.cat_dim = cat_dim
        self.latent_dim = latent_dim
        self.enc_dim = enc_dim

        self.t = gen_param(t)
        self.min_t = gen_param(min_t) 
        self.rate = gen_param(rate)
        self.kl_coeff = gen_param(kl_coeff)
        self.interval = gen_param(interval)

        self.fc_z = nn.Linear(enc_dim, latent_dim * cat_dim)
        self.log_scale = gen_param(0.0)
    
    def update_t(self, batch_idx):
        self.t = torch.nn.Parameter(torch.max(self.t * torch.exp(- self.rate * batch_idx),
                                   self.min_t))
    
    def reparameterize(self, z, eps=1e-20):
        u = torch.rand_like(z)
        g = - torch.log(-torch.log(u + eps) + eps)

        # Gumbel-Softmax Trick
        s = F.softmax((z + g) / self.t, dim=-1)
        s = s.view(-1, self.l_dim * self.c_dim)
        return s

    def kl_divergence(self, q, eps=1e-20):
        q_p = nn.functional.softmax(q, dim=-1)
        e = q_p * torch.log(q_p + eps)
        ce = q_p * np.log(1. / self.cat_dim + eps)

        kl = torch.mean(torch.sum(e - ce, dim =(1,2)), dim=0)
        return kl
    
    @abstractmethod
    def forward(self, *args, training=False):
        raise NotImplementedError
    
    @abstractmethod
    def training_step(self, *args, batch_idx=None):
        raise NotImplementedError

class SequentialVAE(genericVAE):
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
        q = self.fc_z(x_encoded)
        q = q.view(-1, self.cat_dim, self.latent_dim)
        if training:
            q = self.reparameterize(q)
        y = self.decoder(q)

        if training: return q, y
        return y
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, q = self.forward(x, training=True)
        kl = self.kl_divergence(q).mean()
        loss = F.cross_entropy(y_hat, y) + self.kl_coeff * kl
        if batch_idx % self.interval == 0:
            self.update_t(batch_idx)
        return loss

# EnsembleVAE takes a callable head function and an encoder as arguments
class EnsembleVAE(genericVAE):
    agg = {
        'mean' : lambda x : torch.mean(x, dim=0),
        'max' : lambda x : torch.max(x, dim=0)[0], # fix this
    }
    def __init__(self, 
                 encoder,
                 head : callable,
                 num_heads :int = 2,
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
        q = self.fc_z(x_encoded)
        q = q.view(-1, self.cat_dim, self.latent_dim)
        if training: 
            q = [self.reparameterize(q) for i in range(self.num_heads)]
            q_y = [head(v) for head, v in zip(self.heads, q)]
        else:
            q_y = [head(q) for head in self.heads]

        y = self.agg_func(torch.stack(q_y), dim=0)

        if training: return q, q_y, y
        return y
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        q, q_y, _ = self.forward(x, training=True)
        kl = self.kl_divergence(q).mean()
        ce = torch.sum(torch.stack([F.cross_entropy(_y, y) for _y in q_y]), axis=0)
        loss = ce + self.kl_coeff * kl
        if batch_idx % self.interval == 0:
            self.update_t(batch_idx)
        return {'loss' : loss, 'ce' : ce, 'kl' : kl}