import pytorch_lightning as pl
from torch import nn
import torch
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import torchmetrics
import numpy as np


class ensembleVAEclassifier(pl.LightningModule):
    def __init__(self, 
            heads,
            num_heads,
            enc_out_dim=512, 
            latent_dim=10, 
            categorical_dim=10,
            input_height=32, 
            in_channels=3,
            temperature: float = 0.5,
            min_temperature: float = 0.2,
            anneal_rate: float = 3e-5,
            anneal_interval: int = 100, # every 100 batches
            alpha: float = 2.,
            kl_coeff = 0.1,
            **kwargs):
        super().__init__(**kwargs)
        
        gen_param = lambda x : nn.Parameter(torch.Tensor([x]))

        self.save_hyperparameters(ignore='heads')

        self.l_dim = latent_dim
        self.c_dim = categorical_dim
        self.num_head = num_heads

        self.t = gen_param(temperature)
        self.min_t = gen_param(min_temperature)
        self.rate = gen_param(anneal_rate)
        self.interval = gen_param(anneal_interval)
        self.alpha = gen_param(alpha)
        self.kl_coeff = gen_param(kl_coeff)
        
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoders = nn.ModuleList([resnet18_decoder(
                        latent_dim=latent_dim*categorical_dim,
                        input_height=input_height,
                        first_conv=False,
                        maxpool1=False
                        ) for i in range(num_heads)])
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.heads = nn.ModuleList(heads)

        # distribution parameters
        self.fc_z = nn.Linear(enc_out_dim, latent_dim * categorical_dim)
        
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, q, eps=1e-20):
        q_p = nn.functional.softmax(q, dim=-1)
        e = q_p * torch.log(q_p + eps)
        ce = q_p * np.log(1. / self.c_dim + eps)

        kl = torch.mean(torch.sum(e - ce, dim =(1,2)), dim=0)
        return kl
    
    def elbo(self, kl, recons):
        loss = (self.kl_coeff) * kl - self.alpha * recons
        return loss.mean()
    
    def reparameterize(self, z, eps=1e-20):
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax Trick
        s = nn.functional.softmax((z + g) / self.t, dim=-1)
        s = s.view(-1, self.l_dim * self.c_dim)
        return s

    def training_step(self, batch, batch_idx):
        x, y = batch

        # encode x to get gaussian parameters
        x_encoded = self.encoder(x)

        q = self.fc_z(x_encoded)
        q = q.view(-1, self.l_dim, self.c_dim)
        Z = [self.reparameterize(q) for i in range(self.num_head)]
 
        # decoded
        X_hat = [decoder(z) for decoder, z in zip(self.decoders, Z)]
        y_preds = [head(x_hat) for head, x_hat in zip(self.heads, X_hat)]

        if batch_idx % self.interval == 0:
            self.t = torch.nn.Parameter(torch.max(self.t * torch.exp(- self.rate * batch_idx),
                                   self.min_t))

        # reconstruction loss
        recons_loss = torch.stack([self.gaussian_likelihood(x_hat, self.log_scale, x) for x_hat in X_hat])

        # kl
        kl = self.kl_divergence(q)

        label_error = torch.sum(torch.stack([nn.functional.cross_entropy(y_pred, y) for y_pred in y_preds]))

        # elbo
        elbo = torch.sum(torch.stack([self.elbo(kl, recons) for recons in recons_loss]))

        self.accuracy(torch.mean(torch.stack([y_pred for y_pred in y_preds]), axis=0), y)

        self.log_dict({
            'elbo': elbo,
            'kl': -kl.mean(),
            'recon_loss': recons_loss.mean(),
            'cce' : label_error,
            'train_acc_step' : self.accuracy
        })

        return elbo + label_error
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        q = self.fc_z(x_encoded)
        X_hat = [decoder(q) for decoder in self.decoders]
        y_hat = torch.mean(torch.stack([head(x_hat) for head,x_hat in zip(self.heads, X_hat)]), axis=0)

        
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        q = self.fc_z(x_encoded)
        X_hat = [decoder(q) for decoder in self.decoders]
        y_hat = torch.mean(torch.stack([head(x_hat) for head,x_hat in zip(self.heads, X_hat)]), axis=0)
        
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss