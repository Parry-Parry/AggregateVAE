import pytorch_lightning as pl
from torch import nn
import torch
import torchmetrics
import numpy as np

class classifier_head(pl.LightningModule):
    def __init__(self, in_dim, linear_stack, n_class=10, **kwargs):
        super().__init__(**kwargs)
        layers = []
        for size in linear_stack:
            layers.append(
                nn.Sequential(
                        nn.Linear(in_dim, size),
                        nn.ReLU()
                )
            )
            in_dim = size
        layers.append(
            nn.Sequential(
                    nn.Linear(linear_stack[-1], n_class),
                    nn.Softmax()
            )
        )
        self.classifier = nn.Sequential(*layers)
    def forward(self, x):
        return self.classifier(x)

class VAEclassifier(pl.LightningModule):
    def __init__(self, 
            head,
            dim,
            stack,
            latent_dim=10, 
            categorical_dim=10,
            temperature: float = 0.5,
            min_temperature: float = 0.2,
            anneal_rate: float = 3e-5,
            anneal_interval: int = 100, # every 100 batches
            alpha: float = 2.,
            kl_coeff = 0.1,
            **kwargs):
        super().__init__(**kwargs)
        
        gen_param = lambda x : nn.Parameter(torch.Tensor([x]))

        self.save_hyperparameters(ignore='head')

        self.l_dim = latent_dim
        self.c_dim = categorical_dim

        self.t = gen_param(temperature)
        self.min_t = gen_param(min_temperature)
        self.rate = gen_param(anneal_rate)
        self.interval = gen_param(anneal_interval)
        self.alpha = gen_param(alpha)
        self.kl_coeff = gen_param(kl_coeff)

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=categorical_dim)
        self.rec = torchmetrics.Recall(task='multiclass', average='macro', num_classes=categorical_dim)
        self.prec = torchmetrics.Precision(task='multiclass', average='macro', num_classes=categorical_dim)

        in_dim = dim
        layers = []
        for size in stack:
            layers += [
                nn.Linear(in_dim, size),
                nn.ReLU()
            ]
            in_dim = size
        
        layers += [
            nn.Linear(in_dim, latent_dim * categorical_dim),
            nn.ReLU()
        ]

        self.encoder = nn.Sequential(*layers)

        stack = stack[::-1]
        layers = []

        in_dim = latent_dim * categorical_dim
        for size in stack:
            layers += [
                nn.Linear(in_dim, size),
                nn.ReLU()
            ]
            in_dim = size
        layers += [
            nn.Linear(in_dim, dim),
        ]

        self.decoder = nn.Sequential(*layers)
        self.head = head

        
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
    
    def reparameterize(self, z, eps=1e-20):
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax Trick
        s = nn.functional.softmax((z + g) / self.t, dim=-1)
        s = s.view(-1, self.l_dim * self.c_dim)
        return s

    def training_step(self, batch, batch_idx):
        x, y = batch

        # encode x to get the mu and variance parameters
        q = self.encoder(x)
        q = q.view(-1, self.l_dim, self.c_dim)
        z = self.reparameterize(q)

        # decoded
        x_hat = self.decoder(z)

        y_pred = self.head(x_hat)

        if batch_idx % self.interval == 0:
            self.t = torch.nn.Parameter(torch.max(self.t * torch.exp(- self.rate * batch_idx),
                                   self.min_t))

        # reconstruction loss
        recons_loss = self.gaussian_likelihood(x_hat, self.log_scale, x) 

        # kl
        kl = self.kl_divergence(q)

        label_error = nn.functional.cross_entropy(y_pred, y)

        # elbo
        elbo = (self.kl_coeff)*kl - self.alpha * recons_loss
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': -kl.mean(),
            'recon_loss': recons_loss.mean(),
            'cce' : label_error
        })

        return elbo + label_error
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        q = self.fc_z(x_encoded)
        x_hat = self.decoder(q)
        y_hat = self.head(x_hat)
        
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        q = self.fc_z(x_encoded)
        x_hat = self.decoder(q)
        y_hat = self.head(x_hat)
        
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss