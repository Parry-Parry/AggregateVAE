from abc import abstractmethod
import torch 
import torch.nn as nn
import torch.nn.functional as F

gen_param = lambda x : nn.Parameter(torch.Tensor([x]))

class GenericClassifier(nn.Module):
    def __init__(self,
                 device = None,
                 enc_dim : int = 200,
                 loss_fn : callable = F.cross_entropy,
                 latent_dim : int = 200,
                 epsilon : float = 0.1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.enc_dim = enc_dim
        self.loss_fn = loss_fn
        self.fc_z = nn.Linear(enc_dim, latent_dim)

        self.epsilon = epsilon if epsilon else None
        self.forward = self.epsilon_forward if epsilon else self.std_forward
    
    @abstractmethod
    def epsilon_forward(self, x, training=False):
        raise NotImplementedError
    
    @abstractmethod
    def std_forward(self, x, training=False):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.LongTensor) 
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x, training=True)
        loss = self.loss_fn(y_hat, y)
        return {'loss' : loss}
    
    @torch.no_grad()
    def validation_step(self, loader, eval_metrics): 
        for batch in loader:
            x, y = batch
            y = y.type(torch.LongTensor)
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat, y)

            for _, func in eval_metrics.items(): func.update(y_hat.cpu(), y.cpu())
        
        metrics = {m : func.compute() for m, func in eval_metrics.items()}
        for _, func in eval_metrics.items(): func.reset() 
        return {'val_loss' : loss, **metrics}

class SequentialClassifier(GenericClassifier):
    def __init__(self, 
                 encoder, 
                 head,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = encoder
        self.head = head
    
    def epsilon_forward(self, x, training=False):
        if training: x = x + torch.Tensor(x.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        z = self.fc_z(x_encoded)
        y_hat = self.head(z)
        return y_hat
    
    def std_forward(self, x, training=False):
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        z = self.fc_z(x_encoded)
        y_hat = self.head(z)
        return y_hat

class EnsembleClassifier(GenericClassifier):
    agg = {
        'mean' : lambda x : torch.mean(x, dim=0),
        'max' : lambda x : torch.max(x, dim=0)[0], # fix this
    }
    def __init__(self, 
                 encoder,
                 head : callable,
                 num_heads : int = 2,
                 agg : str = 'mean',
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.head = nn.ModuleList([head(i) for i in range(num_heads)])
        self.num_heads = num_heads
        self.agg_func = self.agg[agg]
    
    def epsilon_forward(self, x, training=False):
        if training:
            X = [x for _ in range(self.num_heads)]
            X = map(lambda x : x + torch.Tensor(x.shape).uniform_(-self.epsilon, self.epsilon).to(self.device), X)
            x_encoded = map(self.encoder, X)
            x_encoded = map(lambda x : x.view(x.size(0), -1), x_encoded)
            Z = map(self.fc_z, x_encoded)
            inter_y = torch.stack([head(z) for head, z in zip(self.head, Z)])

        else:
            x_encoded = self.encoder(x)
            z = self.fc_z(x_encoded.view(x_encoded.size(0), -1))
            inter_y = torch.stack([head(z) for head in self.head])
        
        y_hat = self.agg_func(inter_y)
        if training: return y_hat, inter_y
        return y_hat

    def std_forward(self, x, training=False):
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        z = self.fc_z(x_encoded)
        inter_y = torch.stack([head(z) for head in self.head])
        y_hat = self.agg_func(inter_y)
        if training: return y_hat, inter_y
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.LongTensor)
        x, y = x.to(self.device), y.to(self.device)
        _, inter_y = self.forward(x, training=True)
        loss = torch.sum(torch.stack([self.loss_fn(_y, y) for _y in inter_y]))
        return {'loss' : loss}