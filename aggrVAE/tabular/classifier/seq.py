import pytorch_lightning as pl
from torch import nn
import torch
import torchmetrics
import numpy as np

class classifier_head(pl.LightningModule):
    def __init__(self, in_dim, linear_stack, n_class=10, **kwargs):
        super(classifier_head, self).__init__(**kwargs)
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

class Classifier(pl.LightningModule):
    def __init__(self, 
            head,
            dim,
            stack,
            epsilon=0.01,
            latent_dim=10, 
            categorical_dim=10):
        super().__init__()

        self.save_hyperparameters(ignore='head')

        self.l_dim = latent_dim
        self.c_dim = categorical_dim

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
        self.head = head

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.epsilon: x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        q = self.encoder(x)
        y_pred = self.head(q)

        label_error = nn.functional.cross_entropy(y_pred, y.float())

        self.log_dict({
            'cce' : label_error
        })

        return label_error 
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        q = self.encoder(x)
        y_hat = self.head(q)
        
        loss = nn.functional.cross_entropy(y_hat, y.float())
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        q = self.encoder(x)
        y_hat = self.head(q)
        
        loss = nn.functional.cross_entropy(y_hat, y.float())
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