import pytorch_lightning as pl
import torchmetrics
from torch import nn
import torch
import torchvision
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import numpy as np

def compute_conv(input_vol, stack, kernel_size, stride, padding):
    vol = input_vol
    for i in range(len(stack)):
        vol = ((vol - kernel_size + 2 * padding) / stride) + 1
    return int(vol * vol * stack[-1])

class classifier_head(pl.LightningModule):
    def __init__(self, encoder, linear_stack, n_class=10, **kwargs):
        super().__init__(**kwargs)
        layers = []
        in_dim = 512

        self.encoder = encoder

        layers.append(nn.Flatten())

        for size in linear_stack:
            layers.append(
                nn.Sequential(
                        nn.Linear(in_dim, size),
                        nn.Softmax()
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
        x = self.encoder(x)
        return self.classifier(x)   

class resnetClassifier(pl.LightningModule):
    def __init__(self, head, categorical_dim=10, in_channels=3):
        super().__init__()

        self.prepare_data_per_node = False
        self.save_hyperparameters(ignore='head')

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.head = head

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # encode x to get the mu and variance parameters
        z = self.encoder(x)
        y_pred = self.head(z)

        label_error = nn.functional.cross_entropy(y_pred, y.long())

        self.accuracy(y_pred, y.long())

        self.log_dict({
            'cce' : label_error,
            'train_acc_step' : self.accuracy
        })

        return label_error
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        y_pred = self.head(z)

        loss = nn.functional.cross_entropy(y_pred, y.long())
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        y_pred = self.head(z)
        
        loss = nn.functional.cross_entropy(y_pred, y)
        self.test_accuracy(y_pred, y.long())

        self.log_dict({
            'test_loss' : loss,
            'test_acc_step' : self.test_accuracy
        })
        return loss