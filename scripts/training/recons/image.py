import os
from fire import Fire
import multiprocessing as mp
import torch
from os.path import join
from aggrVAE.models.classifier import SequentialClassifier, EnsembleClassifier
from aggrVAE.models.modules import ConvEncoder
from aggrVAE.datamodule import ReconsCIFAR10DataModule, ReconsMNISTDataModule
from aggrVAE.util import callable_head, LogStore, Log, init_out, dump_logs

STACK = [512, 256, 128]

cpus = mp.cpu_count()

ds_funcs = {
    'mnist' : ReconsMNISTDataModule,
    'cifar10' : ReconsCIFAR10DataModule,
}

def main(dataset : str, 
         datastore : str, 
         outstore : str,
         trainstore : str = None,
         num_heads : int = 1,
         epochs : int = 1,
         batch_size : int = 128,
         epsilon : float = None,
         enc_dim : int = 512,
         latent_dim : int = 10,
         cat_dim : int = 10,
         p : int = 5,
         gpus=0):
    
    init_out(outstore)
    store = LogStore([], {})

    ds = ds_funcs[dataset](trainstore, batch_size, cpus, datastore, epsilon, p)

    ds.prepare_data()
    ds.setup()

    encoder = ConvEncoder(in_channels=ds.channels)
    head = callable_head(latent_dim * cat_dim, ds.classes)
    
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
        log = Log(epoch, {}, {})
        error = []
        for batch_idx, batch in enumerate(train):
            optimizer.zero_grad() 
            loss = model.training_step(batch, batch_idx)
            error.append(loss)
            loss['loss'].backward()
            optimizer.step()
        validation = model.validation_step(val)
        print(f'Epoch {epoch} : {validation}')

        log.loss.extend({k : sum([e[k] for e in error])/len(error) for k in error[0].keys()})
        log.val_metrics.extend(validation)
        store.logs.append(log)
    
    test = model.validation_step(test)
    store.test_metrics.extend(test)

    torch.save(model.state_dict(), join(outstore, 'models', f'{dataset}.{epochs}.model.{num_heads}.recons.pt'))
    dump_logs(store, os.path.join(outstore, 'logs.json'))   

if __name__ == '__main__':
    Fire(main)