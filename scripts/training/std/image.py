import os
import logging
from tqdm.auto import tqdm
from fire import Fire
import multiprocessing as mp
import torch
import torchmetrics
from os.path import join
from aggrVAE.models.vae import SequentialVAE, EnsembleVAE
from aggrVAE.models.classifier import SequentialClassifier, EnsembleClassifier
from aggrVAE.models.modules import ConvEncoder
from aggrVAE.datamodule import MNISTDataModule, CIFAR10DataModule, AggrCIFAR10DataModule, AggrMNISTDataModule
from aggrVAE.util import callable_head, LogStore, Log, init_out, dump_logs

STACK = [512, 256, 128]


cpus = mp.cpu_count()

ds_funcs = {
    'mnist' : MNISTDataModule,
    'cifar10' : CIFAR10DataModule,
    'aggrmnist' : AggrMNISTDataModule,
    'aggrcifar10' : AggrCIFAR10DataModule
}

def main(dataset : str, 
         datastore : str, 
         outstore : str,
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
         epsilon : float = None,
         gpus=0):
    
    init_out(outstore)
    store = LogStore([], {})

    if trainstore: 
        assert trainstore is not None
        ds = ds_funcs[f'aggr{dataset}'](trainstore, batch_size, cpus, datastore)
    else: ds = ds_funcs[dataset](batch_size, cpus, datastore)

    device = 'cuda' if gpus > 0 else 'cpu'
    device = torch.device(device)

    ds.prepare_data()
    ds.setup()

    metrics = {'accuracy' : torchmetrics.Accuracy(task="multiclass", num_classes=ds.classes), 
           'f1' : torchmetrics.F1Score(task="multiclass", num_classes=ds.classes),
           'precision' : torchmetrics.Precision(task="multiclass", num_classes=ds.classes),
           'recall' : torchmetrics.Recall(task="multiclass", num_classes=ds.classes)}

    encoder = ConvEncoder(in_channels=ds.channels)
    head = callable_head(latent_dim * cat_dim, STACK, ds.classes)

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
                        interval=interval,
                        device=device)
        else:
            model = SequentialVAE(encoder,
                            head(),
                            enc_dim=enc_dim,
                            latent_dim=latent_dim,
                            cat_dim=cat_dim,
                            kl_coeff=kl_coeff,
                            interval=interval,
                            device=device)
    else:
        if num_heads > 1: 
            model = EnsembleClassifier(encoder,
                                        head,
                                        num_heads,
                                        'mean',
                                        enc_dim=enc_dim,
                                        latent_dim=cat_dim*latent_dim,
                                        epsilon=epsilon,
                                        device=device)
        else:
            model = SequentialClassifier(encoder,
                                         head(),
                                         enc_dim=enc_dim,
                                         latent_dim=cat_dim*latent_dim,
                                         epsilon=epsilon,
                                         device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model = model.to(device)

    train = ds.train_dataloader()
    val = ds.val_dataloader()
    test = ds.test_dataloader()

    for epoch in tqdm(range(epochs)):
        log = Log(epoch, {}, {})
        error = []
        for batch_idx, batch in enumerate(train):
            optimizer.zero_grad() 
            loss = model.training_step(batch, batch_idx)
            error.append({str(k) : v.detach().cpu().numpy() for k, v in loss.items()})
            loss = loss['loss']
            loss.backward()
            optimizer.step()

        validation = model.validation_step(val, metrics)
        logging.info(f'Epoch {epoch} : {validation}')
        log.loss.update({epoch : {str(k) : sum([e[k].detach().cpu().numpy() for e in error])/len(error) for k in error[0].keys()}})
        loss = log.loss[epoch]['loss']
        logging.info(f'Epoch {epoch} : Loss: {loss}')
        log.val_metrics.update(validation)
        store.logs.append(log)
    
    test = model.validation_step(test, metrics)
    store.test_metrics.update({str(k) : v.detach().cpu().numpy() for k, v in test.items()})
    logging.info(store)
    vae = 'vae' if vae else 'std'
    torch.save(model.state_dict(), join(outstore, 'models', f'{dataset}.{epochs}.model.{num_heads}.{vae}.pt'))
    dump_logs(store, outstore)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)