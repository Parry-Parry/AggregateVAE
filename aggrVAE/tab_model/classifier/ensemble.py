import pytorch_lightning as pl
from torch import nn
import torch
import torchmetrics
import numpy as np


class EnsembleClassifier(pl.LightningModule):
    def __init__(self, 
            heads,
            stack,
            dim,
            num_heads,
            epsilon=0.01,
            latent_dim=10, 
            categorical_dim=10):
        super().__init__()

        self.save_hyperparameters(ignore='heads')

        self.l_dim = latent_dim
        self.c_dim = categorical_dim
        self.num_head = num_heads

        if epsilon != 0: self.epsilon = epsilon
        else: self.epsilon = None

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
        self.heads = nn.ModuleList(heads)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.epsilon : X = [x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon) for i in range(self.num_head)]
        else: X = [x for i in range(self.num_head)]
        X_encoded = [self.encoder(x_hat) for x_hat in X]
    
        y_preds = [head(z) for head, z in zip(self.heads, X_encoded)]

        label_error = torch.sum(torch.stack([nn.functional.cross_entropy(y_pred, y) for y_pred in y_preds]))
        y_hat = torch.mean(torch.stack([y_pred for y_pred in y_preds]), axis=0)

        self.train_acc(y_hat, y)

        self.log_dict({
            'cce' : label_error,
            'train_acc_step' : self.train_acc
        })

        return label_error
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        y_hat = torch.mean(torch.stack([head(x_encoded) for head in self.heads]), axis=0)

        loss = nn.functional.cross_entropy(y_hat, y.long())
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        y_hat = torch.mean(torch.stack([head(x_encoded) for head in self.heads]), axis=0)

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