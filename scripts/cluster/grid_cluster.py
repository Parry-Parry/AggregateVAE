from fire import Fire
from typing import List
from os.path import join
from subprocess import run
K = [50, 100, 200, 500, 1000]

def main(script : str,
         dataset : str, 
         datastore : str, 
         outdir : str, 
         n_class : int = 10, 
         verbose : bool = False, 
         seed : int = 42, 
         gpu : bool = False, 
         target : str = None, 
         categorical : List[str] =None):
    args = [
        'python', script,
        '--dataset', dataset,
        '--datastore', join(datastore,dataset),
        '--n_class', str(n_class),
    ]

    for k in K:
        args.extend(['--outdir', join(outdir, f'{dataset}.{k}.npy')])
        if verbose: args.append('--verbose')
        if seed: args.extend(['--seed', str(seed)])
        if gpu: args.append('--gpu')
        args.extend(['--K', str(k)])
        if target: args.extend(['--target', target])
        if categorical: args.extend(['--categorical', categorical])
        run(args)

if __name__ == '__main__':
    Fire(main)