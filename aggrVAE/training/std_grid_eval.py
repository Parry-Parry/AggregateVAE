from fire import Fire 
import os
from os.path import join

HEADS = [1, 3, 5, 10, 20]
EPOCHS = [10, 20, 30, 40, 50]
EPSILON = [0.0001, 0.005, 0.01, 0.05, 0.1, 0.5]

def main(script : str, 
         dataset : str,
         datastore : str, 
         outstore : str, 
         trainstore : str = None,  
         batch_size : int = 8, 
         latent_dim : int = 10, 
         cat_dim : int = 10, 
         enc_dim : int = 512, 
         vae : bool = False,
         p : int = None, 
         gpus=0):
    
    args = ['python', 
            script, 
            '--dataset', dataset,
            '--datastore', datastore,
            '--batch_size', str(batch_size),
            '--latent_dim', str(latent_dim),
            '--cat_dim', str(cat_dim),
            '--enc_dim', str(enc_dim),
            '--gpus', str(gpus)]

    if trainstore: args.extend(['--trainstore', trainstore])
    if vae: args.append('--vae')
    if p: args.extend(['--p', str(p)])

    for epoch in EPOCHS:
        args.extend(['--epochs', str(epoch)])
        for head in HEADS:
            args.extend(['--num_heads', str(head)])
            for epsilon in EPSILON:
                vae_str = 'vae' if vae else 'classifier'
                args.extend('--outstore', join(outstore, f'{dataset}-{epoch}-{head}-{epsilon}-{vae_str}'))
                args.extend(['--epsilon', str(epsilon)])
                print(' '.join(args))
                os.system(' '.join(args))
if __name__ == '__main__':
    Fire(main)