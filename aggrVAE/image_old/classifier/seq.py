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

class classifier_head(pl.LightningModule):
    def __init__(self, linear_stack, encoder=None, n_class=10, dim=200):
        super().__init__()
        layers = []
        in_dim = dim

        if encoder:
            self.encoder = encoder
            layers.append(nn.Flatten())
        else:
            self.encoder = None

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
        if self.encoder: x = self.encoder(x)
        return self.classifier(x)   

class Classifier(pl.LightningModule):
    def __init__(self, 
            linear_stack,
            epsilon=0.01,
            latent_dim=10, 
            categorical_dim=10,
            in_channels=3):
        super().__init__()

        self.prepare_data_per_node = False
        self.save_hyperparameters(ignore='head')
        
        self.l_dim = latent_dim
        self.c_dim = categorical_dim

        if epsilon != 0: self.epsilon = epsilon
        else: self.epsilon = None
            
        self.loss = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=categorical_dim)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=categorical_dim)
        self.rec = torchmetrics.Recall(task='multiclass', average='macro', num_classes=categorical_dim)
        self.prec = torchmetrics.Precision(task='multiclass', average='macro', num_classes=categorical_dim)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding='same'),
            nn.ReLU()
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU()
            nn.MaxPool2d(2),
            nn.Dropout(0.25), 
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU()
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU()
            nn.MaxPool2d(2),
            nn.Dropout(0.25), 
            nn.Flatten()
        )
        
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

        self.fc_z = nn.Linear(512, latent_dim * categorical_dim)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.epsilon: x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
    
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x.shape[0], -1)
        
        q = self.fc_z(x_encoded)
        y_hat = self.head(q)
        
        label_error = self.loss(y_hat, y.long())

        self.train_acc(y_hat, y.float())

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
        y_hat = self.head(x_scale)
        
        loss = nn.functional.cross_entropy(y_hat, y.long())
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x.shape[0], -1)
        x_scale = self.fc_z(x_encoded)
        y_hat = self.head(x_scale)
        
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