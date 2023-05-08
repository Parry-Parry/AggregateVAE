from fire import Fire 
import os
from os.path import join

HEADS = [1, 3, 5, 10, 20]
EPOCHS = [50]
EPSILON = [0., 0.001, 0.005, 0.01, 0.05, 0.1]

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
         gpus=0,
         eps : float = None):
    
    args = ['python', 
            script, 
            '--dataset', dataset,
            '--datastore', datastore,
            '--batch_size', str(batch_size),
            '--latent_dim', str(latent_dim),
            '--cat_dim', str(cat_dim),
            '--enc_dim', str(enc_dim),
            '--gpus', str(gpus)]
    
    vae_str = 'vae' if vae else 'classifier'

    if trainstore: args.extend(['--trainstore', trainstore])
    if vae: args.append('--vae')
    if p: args.extend(['--p', str(p)])
    
    for epoch in EPOCHS:
        for head in HEADS:
            tmp_args = args.copy()
            tmp_args.extend(['--epochs', str(epoch)])
            tmp_args.extend(['--num_heads', str(head)])
            if eps is not None: 
                tmp_args.extend(['--epsilon', str(eps)])
                tmp_args.extend(['--outstore', join(outstore, f'{dataset}-{epoch}-{head}-{eps}-{vae_str}')])
                print(' '.join(tmp_args))
                os.system(' '.join(tmp_args))
            else: 
                for epsilon in EPSILON:
                    eps_args = tmp_args.copy()
                    eps_args.extend(['--outstore', join(outstore, f'{dataset}-{epoch}-{head}-{epsilon}-{vae_str}')])
                    eps_args.extend(['--epsilon', str(epsilon)])
                    print(' '.join(eps_args))
                    os.system(' '.join(eps_args))
if __name__ == '__main__':
    Fire(main)