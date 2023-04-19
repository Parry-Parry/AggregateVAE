from fire import Fire
import multiprocessing as mp
import torch
from os.path import join
from ..models.vae import SequentialVAE, EnsembleVAE
from ..models.classifier import SequentialClassifier, EnsembleClassifier
from ..models.modules import ConvEncoder
from ..datamodule import MNISTDataModule, CIFAR10DataModule, AggrCIFAR10DataModule, AggrMNISTDataModule
from ..util import callable_head

STACK = []

cpus = mp.cpu_count()

ds_funcs = {
    'mnist' : MNISTDataModule,
    'cifar10' : CIFAR10DataModule,
    'aggrmnist' : AggrMNISTDataModule,
    'aggrcifar10' : AggrCIFAR10DataModule
}

def main(dataset : str, 
         datastore : str, 
         aggregate : bool = False,
         trainstore : str = None,
         num_heads : int = 1,
         epochs : int = 1,
         batch_size : int = 128,
         vae : bool = False,
         latent_dim : int = 10,
         cat_dim : int = 10,
         enc_dim : int = 512,
         kl_coeff : float = 1.0,
         interval : int = 100,
         gpus=0):
    

    if aggregate: 
        assert trainstore is not None
        ds = ds_funcs[f'aggr{dataset}'](trainstore, batch_size, cpus, datastore)
    else: ds = ds_funcs[dataset](batch_size, cpus, datastore)

    ds.prepare_data()
    ds.setup()

    encoder = ConvEncoder(in_channels=ds.channels)
    head = callable_head(latent_dim * cat_dim, ds.classes)

    if vae: 
        if num_heads > 1: 
            model = EnsembleVAE(encoder, 
                        head, 
                        num_heads, 
                        'mean', 
                        enc_dim=enc_dim, 
                        latent_dim=latent_dim, 
                        cat_dim=cat_dim, 
                        kl_coeff=kl_coeff, 
                        interval=interval)
        else:
            model = SequentialVAE(encoder,
                            head(),
                            enc_dim=enc_dim,
                            latent_dim=latent_dim,
                            cat_dim=cat_dim,
                            kl_coeff=kl_coeff,
                            interval=interval)
    else:
        if num_heads > 1: 
            model = EnsembleClassifier(encoder,
                                        head,
                                        num_heads,
                                        'mean',
                                        enc_dim=enc_dim,
                                        latent_dim=cat_dim*latent_dim)
        else:
            model = SequentialClassifier(encoder,
                                         head(),
                                         enc_dim=enc_dim,
                                         latent_dim=cat_dim*latent_dim)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if gpus > 0: model = model.cuda()

    train = ds.train_dataloader()
    val = ds.val_dataloader()
    test = ds.test_dataloader()

    if gpus > 0: train = train.cuda()

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train):
            optimizer.zero_grad() 
            loss = model.training_step(batch, batch_idx)
            loss['loss'].backward()
            optimizer.step()
        for batch_idx, batch in enumerate(val):
            loss = model(batch, batch_idx)

if __name__ == '__main__':
    Fire(main)