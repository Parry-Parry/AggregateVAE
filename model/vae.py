import pytorch_lightning as pl
from torch import nn
import torch
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)


class VAEclassifier(pl.Module):
    def __init__(self, 
            head,
            enc_out_dim=512, 
            latent_dim=10, 
            categorical_dim=10,
            input_height=32, 
            temperature: float = 0.5,
            anneal_rate: float = 3e-5,
            anneal_interval: int = 100, # every 100 batches
            alpha: float = 30.,
            kl_coeff = 0.1,
            **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.l_dim = latent_dim
        self.c_dim = categorical_dim

        self.t = temperature
        self.min_t = temperature
        self.rate = anneal_rate
        self.interval = anneal_interval
        self.alpha = alpha
        self.kl_coeff = kl_coeff

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
                        latent_dim=latent_dim * categorical_dim,
                        input_height=input_height,
                        first_conv=False,
                        maxpool1=False
                        )
        self.head = head

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
        ce = q_p * torch.log(1. / self.categorical_dim + eps)

        kl = torch.mean(torch.sum(e - ce, dim =(1,2)), dim=0)
        return kl
    
    def reparameterize(self, z, eps=1e-20):
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax Trick
        s = nn.functional.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.l_dim * self.c_dim)
        return s

    def training_step(self, batch, batch_idx):
        x, y = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        
        q = self.fc_z(x_encoded)
        q = q.view(-1, self.l_dim, self.c_dim)
        z = self.reparameterize(q)

        # decoded
        x_hat = self.decoder(z)

        y_pred = self.head(x_hat)

        if batch_idx % self.interval == 0 and self.training:
            self.t = torch.max(self.t * torch.exp(- self.rate * batch_idx),
                                   self.min_t)

        # reconstruction loss
        recons_loss = self.gaussian_likelihood(x_hat, self.log_scale, x) 

        # kl
        kl = self.kl_divergence(q)

        label_error = nn.functional.cross_entropy(y, y_pred)

        # elbo
        elbo = (self.kl_coeff)*kl - self.alpha * recons_loss
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recons_loss.mean(),
            'cce' : label_error
        })

        return elbo + label_error