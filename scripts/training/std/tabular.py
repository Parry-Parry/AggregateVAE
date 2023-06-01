import logging
import os
from fire import Fire
import multiprocessing as mp
import torch
import torchmetrics 
from os.path import join
from aggrVAE.models.vae import SequentialVAE, EnsembleVAE
from aggrVAE.models.classifier import SequentialClassifier, EnsembleClassifier
from aggrVAE.models.modules import DenseEncoder
from aggrVAE.datamodule import TabularDataModule
from aggrVAE.util import callable_head, LogStore, Log, init_out, dump_logs


STACK = [512, 256, 128]
ENCODER_STACK = [512, 256]

cpus = mp.cpu_count()


def main(dataset : str, 
         datastore : str, 
         outstore : str,
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

    ds = TabularDataModule(batch_size, cpus, datastore)


    ds.prepare_data()
    ds.setup()

    metrics = {'accuracy' : torchmetrics.Accuracy(task="multiclass", num_classes=ds.classes), 
           'f1' : torchmetrics.F1Score(task="multiclass", num_classes=ds.classes),
           'precision' : torchmetrics.Precision(task="multiclass", num_classes=ds.classes),
           'recall' : torchmetrics.Recall(task="multiclass", num_classes=ds.classes)}

    encoder = DenseEncoder(ds.features, ENCODER_STACK, latent_dim=enc_dim)
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
                                        latent_dim=cat_dim*latent_dim,
                                        epsilon=epsilon)
        else:
            model = SequentialClassifier(encoder,
                                         head(),
                                         enc_dim=enc_dim,
                                         latent_dim=cat_dim*latent_dim,
                                         epsilon=epsilon)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if gpus > 0: model = model.cuda()

    train = ds.train_dataloader()
    val = ds.val_dataloader()
    test = ds.test_dataloader()

    for epoch in range(epochs):
        log = Log(epoch, {}, {})
        error = []
        for batch_idx, batch in enumerate(train):
            optimizer.zero_grad() 
            loss = model.training_step(batch, batch_idx)
            error.append({str(k) : v.detach().cpu().numpy().item() for k, v in loss.items()})
            loss = loss['loss']
            loss.backward()
            optimizer.step()

        validation = model.validation_step(val, metrics)
        logging.info(f'Epoch {epoch} : {validation}')
        log.loss.update({epoch : {str(k) : sum([e[k] for e in error])/len(error) for k in error[0].keys()}})
        loss = log.loss[epoch]['loss']
        logging.info(f'Epoch {epoch} : Loss: {loss}')
        log.val_metrics.update({str(k) : v.detach().cpu().numpy().item() for k, v in validation.items()})
        store.logs.append(log)
    
    test = model.validation_step(test, metrics)
    store.test_metrics.update({str(k) : v.detach().cpu().numpy().item() for k, v in test.items()})

    vae = 'vae' if vae else 'std'
    torch.save(model.state_dict(), join(outstore, 'models', f'{dataset}.{epochs}.model.{num_heads}.{vae}.pt'))
    dump_logs(store, outstore)

if __name__ == '__main__':
    Fire(main)