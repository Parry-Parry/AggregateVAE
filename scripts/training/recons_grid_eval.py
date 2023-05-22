from fire import Fire 
import os
from os.path import join

HEADS = [1, 3, 5, 10, 20]
EPOCHS = [50]
EPSILON = [0., 0.001, 0.005, 0.01, 0.05, 0.1]
K = [50, 100, 200, 500, 1000]

def main(script : str, 
         dataset : str,
         datastore : str, 
         outstore : str, 
         trainstore : str, 
         batch_size : int = 8, 
         latent_dim : int = 10, 
         cat_dim : int = 10, 
         enc_dim : int = 512, 
         vae : bool = False,
         gpus : int = 0,
         eps : float = None):
    
    main_args = ['python', 
            script, 
            '--dataset', dataset,
            '--datastore', datastore,
            '--batch_size', str(batch_size),
            '--latent_dim', str(latent_dim),
            '--cat_dim', str(cat_dim),
            '--enc_dim', str(enc_dim),
            '--gpus', str(gpus)]
    
    vae_str = 'vae' if vae else 'classifier'
    
    for epoch in EPOCHS:
        for head in HEADS:
            for k in K:
                args = main_args.copy()
                args.extend(['--trainstore', join(trainstore, f'{dataset}.{k}.npy.npz')])
                tmp_args = args.copy()
                tmp_args.extend(['--epochs', str(epoch)])
                tmp_args.extend(['--num_heads', str(head)])
                if eps is not None: 
                    tmp_args.extend(['--epsilon', str(eps)])
                    tmp_args.extend(['--outstore', join(outstore, f'recons-{k}-{dataset}-{epoch}-{head}-{eps}-{vae_str}')])
                    print(' '.join(tmp_args))
                    os.system(' '.join(tmp_args))
                else: 
                    for epsilon in EPSILON:
                        eps_args = tmp_args.copy()
                        eps_args.extend(['--outstore', join(outstore, f'recons-{k}-{dataset}-{epoch}-{head}-{epsilon}-{vae_str}')])
                        eps_args.extend(['--epsilon', str(epsilon)])
                        print(' '.join(eps_args))
                        os.system(' '.join(eps_args))
if __name__ == '__main__':
    Fire(main)