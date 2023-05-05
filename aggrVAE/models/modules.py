import torch.nn as nn
from typing import List

class Head(nn.Module):
    def __init__(self, in_dim : int, stack : List[int], n_class : int , i=0, **kwargs):
        super().__init__(**kwargs)
        self.name = f'head_{i}'
        layers = []
        for dim in stack:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, dim),
                    nn.ReLU()
                )
            )
            in_dim = dim
        layers.append(
            nn.Sequential(
                nn.Linear(in_dim, n_class),
                nn.Softmax(dim=1)
            )
        )
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class DenseEncoder(nn.Module):
    def __init__(self, in_dim : int, stack : List[int], latent_dim : int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = 'encoder'
        layers = []
        for dim in stack:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, dim),
                    nn.tanh()
                )
            )
            in_dim = dim
        layers.append(
            nn.Linear(in_dim, latent_dim)
        )
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class ConvEncoder(nn.Module):
    def __init__(self, in_channels : int = 3) -> None:
        from torchvision.models import resnet18
        super().__init__()
        self.name = 'encoder'
        resnet = resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(resnet.children())[:-1]
        
        self.layers = nn.Sequential(*modules)
    def forward(self, x):
        return self.layers(x)
