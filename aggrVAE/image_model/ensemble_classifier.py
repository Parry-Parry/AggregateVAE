import pytorch_lightning as pl
import torchmetrics
from torch import nn
import torch
import torchvision
from pl_bolts.models.autoencoders.components import resnet18_encoder
import numpy as np

class resnetClassifier(pl.LightningModule):
    def __init__(self, heads, categorical_dim=10, in_channels=3):
        super().__init__()

        self.prepare_data_per_node = False
        self.save_hyperparameters(ignore='head')

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.head = heads

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # encode x to get the mu and variance parameters
        z = self.encoder(x)
        y_preds = [head(z) for head in self.heads]

        label_error = torch.sum(torch.stack([nn.functional.cross_entropy(y_pred, y) for y_pred in y_preds]))

        self.accuracy(torch.mean(torch.stack([y_pred for y_pred in y_preds]), axis=0), y.long())

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
        y_preds = [head(z) for head in self.heads]
        
        loss = torch.sum(torch.stack([nn.functional.cross_entropy(y_pred, y) for y_pred in y_preds]))
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        y_preds = [head(z) for head in self.heads]
        
        loss = torch.sum(torch.stack([nn.functional.cross_entropy(y_pred, y) for y_pred in y_preds]))
        self.test_accuracy(torch.mean(torch.stack([y_pred for y_pred in y_preds]), axis=0), y.long())

        self.log_dict({
            'test_loss' : loss,
            'test_acc_step' : self.test_accuracy
        })
        return loss