import pytorch_lightning as pl
import torchmetrics
from torch import nn
import torch
import torchvision
from torchvision.models import resnet18
import numpy as np

def compute_conv(input_vol, stack, kernel_size, stride, padding):
    vol = input_vol
    for i in range(len(stack)):
        vol = ((vol - kernel_size + 2 * padding) / stride) + 1
    return int(vol * vol * stack[-1])

class EncoderClassifier(pl.LightningModule):
    def __init__(self, 
            linear_stack,
            enc_out_dim=512, 
            latent_dim=10, 
            categorical_dim=10,
            in_channels=3,
            temperature: float = 0.5,
            min_temperature: float = 0.2,
            anneal_rate: float = 3e-5,
            anneal_interval: int = 100, # every 100 batches
            kl_coeff = 1.):
        super().__init__()
        
        gen_param = lambda x : nn.Parameter(torch.Tensor([x]))

        self.prepare_data_per_node = False
        self.save_hyperparameters(ignore='head')

        self.l_dim = latent_dim
        self.c_dim = categorical_dim

        self.t = gen_param(temperature)
        self.min_t = gen_param(min_temperature)
        self.rate = gen_param(anneal_rate)
        self.interval = gen_param(anneal_interval)
        self.kl_coeff = gen_param(kl_coeff)

        self.loss = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=categorical_dim)
        self.rec = torchmetrics.Recall(task='multiclass', average='macro', num_classes=categorical_dim)
        self.prec = torchmetrics.Precision(task='multiclass', average='macro', num_classes=categorical_dim)

        encoder = resnet18(weights=None)
        encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(encoder.children())[:-1]
        
        self.encoder = nn.Sequential(*modules)

        '''
        for para in self.encoder.parameters():
            para.requires_grad = False
        '''
        
        in_dim = latent_dim * categorical_dim
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
                    nn.Linear(linear_stack[-1], categorical_dim),
                    nn.Softmax()
            )
        )

        self.head = nn.Sequential(*layers)

        # distribution parameters
        self.fc_z = nn.Linear(enc_out_dim, latent_dim * categorical_dim)
        
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
        
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        q = self.fc_z(x_encoded)
        q = q.view(-1, self.l_dim, self.c_dim)
        z = self.reparameterize(q)

        y_pred = self.head(z)

        if batch_idx % self.interval == 0:
            self.t = torch.nn.Parameter(torch.max(self.t * torch.exp(- self.rate * batch_idx),
                                   self.min_t))

        # kl
        kl = self.kl_divergence(q).mean()

        label_error = self.loss(y_pred, y.long())

        self.train_acc(y_pred, y.long())

        self.log_dict({
            'kl': -kl,
            'cce' : label_error,
            'train_acc_step' : self.train_acc
        })

        return label_error 
    
    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_acc)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        q = self.fc_z(x_encoded)
        y_hat = self.head(q)
        
        loss = nn.functional.cross_entropy(y_hat, y.long())
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        q = self.fc_z(x_encoded)
        y_hat = self.head(q)
        
        loss = nn.functional.cross_entropy(y_hat, y.long())
        self.acc(y_hat, y.long())
        self.f1(y_hat, y.long())
        self.rec(y_hat, y.long())
        self.prec(y_hat, y.long())

        self.log("test_loss", loss)
        return loss
    
    def test_epoch_end(self, outs):
        self.log_dict(
            {
                'test_acc' : self.acc,
                'f1' : self.f1,
                'recall' : self.rec, 
                'precision' : self.prec
            }
        )
