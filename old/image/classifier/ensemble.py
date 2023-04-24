import pytorch_lightning as pl
from torch import nn
import torch
from torchvision.models import resnet18

import torchmetrics
import numpy as np


class EnsembleClassifier(pl.LightningModule):
    def __init__(self, 
            linear_stack,
            num_heads,
            epsilon=0.01,
            latent_dim=10, 
            categorical_dim=10,
            in_channels=3):
        super().__init__()
        
        self.save_hyperparameters(ignore='heads')

        self.l_dim = latent_dim
        self.c_dim = categorical_dim
        self.num_head = num_heads

        self.loss = nn.CrossEntropyLoss()
        
        if epsilon != 0: self.epsilon = epsilon
        else: self.epsilon = None
      
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

        self.heads = nn.ModuleList([nn.Sequential(*layers) for i in range(num_heads)])

        self.fc_z = nn.Linear(512, latent_dim * categorical_dim)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def kl_divergence(self, q, eps=1e-20):
        q_p = nn.functional.softmax(q, dim=-1)
        e = q_p * torch.log(q_p + eps)
        ce = q_p * np.log(1. / self.c_dim + eps)

        kl = torch.mean(torch.sum(e - ce, dim =(1,2)), dim=0)
        return kl

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.epsilon : X = [x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon) for i in range(self.num_head)]
        else: X = [x for i in range(self.num_head)]
        X_encoded = [self.encoder(x_hat) for x_hat in X]
        X_encoded = [x_enc.view(x_enc.shape[0], -1) for x_enc in X_encoded]
        X_scaled = [self.fc_z(x_enc) for x_enc in X_encoded]
    
        y_preds = [head(z) for head, z in zip(self.heads, X_scaled)]

        label_error = torch.sum(torch.stack([self.loss(y_pred, y.long()) for y_pred in y_preds]))
        y_hat = torch.mean(torch.stack([y_pred for y_pred in y_preds]), axis=0)

        self.train_acc(y_hat, y)

        self.log_dict({
            'cce' : label_error,
            'train_acc_step' : self.train_acc
        })

        return label_error
    
    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_acc)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x.shape[0], -1)
        x_scale = self.fc_z(x_encoded)
        y_hat = torch.mean(torch.stack([head(x_scale) for head in self.heads]), axis=0)

        loss = nn.functional.cross_entropy(y_hat, y.long())
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x.shape[0], -1)
        x_scale = self.fc_z(x_encoded)
        y_hat = torch.mean(torch.stack([head(x_scale) for head in self.heads]), axis=0)

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