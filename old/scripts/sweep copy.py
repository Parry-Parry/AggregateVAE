from argparse import ArgumentParser
import logging
import os
import subprocess

scripts = {
    'image' : {
        'std' : 'train_image.py',
        'aggr' : 'train_image_aggr.py',
        'aggr_enc': 'train_image_aggr_enc.py',
    },
    'tabular' : {
        'std' : 'train_tabular.py',
        'aggr' : 'train_tabular_aggr.py',
        'aggr_enc': 'train_tabular_aggr_enc.py',
    }
}

params = {
    'image' : {
        'K' : [50, 100, 200, 500, 1000],
        'heads' : [1, 3, 5, 10, 20],
        'epsilon' : [0.001, 0.005, 0.01, 0.05, 0.1]
    },
    'tabular' : {
        'heads' : [1, 3, 5, 10, 20],
        'epsilon' : [0.001, 0.005, 0.01, 0.05, 0.1],
        'variations' : ['sex', 'sex_age', 'weight']
    }
}


parser = ArgumentParser()

parser.add_argument('-script_dir', type=str)
parser.add_argument('-data_sink', type=str)
parser.add_argument('-log_dir', type=str)
parser.add_argument('-data', type=str)
parser.add_argument('-model', type=str)
parser.add_argument('--ds', type=str, default='CIFAR10')
parser.add_argument('--data_source', type=str, default=None)
parser.add_argument('--epochs', type=str, default='30')
parser.add_argument('--batch_size', type=str, default='32')
parser.add_argument('--num_workers', type=str, default='4')

parser.add_argument('--verbose', action='store_true')

def main(args):
    parameters = params[args.data]
    script = os.path.join(args.script_dir, scripts[args.data][args.model])
    if 'aggr' in args.model and args.data == 'image': assert args.data_source is not None
    source = args.data_source if args.data_source else 'NA'

    construct = ['python', 
                script, 
                '-data_sink',
                args.data_sink,
                '-log_dir',
                args.log_dir]
    if args.data == 'image': construct.extend(['-ds', args.ds])
    construct.extend(['--data_source',
                source,
                '--epochs',
                args.epochs,
                '--batch_size',
                args.batch_size,
                '--num_workers',
                args.num_workers])

    if args.data == 'tabular':
        for v in parameters['variations']:
            for h in parameters['heads']:
                cmd = construct.copy()
                cmd.extend([
                    '--heads',
                    str(h),
                    '--variant',
                    v
                ])
                if args.model == 'aggr':
                    for e in parameters['epsilon']:
                        cmd.extend(['--epsilon', str(e)])
                        subprocess.run(cmd)
                else:
                    cmd.extend(['--epsilon', '0'])
                    subprocess.run(cmd)
    else:
        for k in parameters['K']:
            for h in parameters['heads']:
                cmd = construct.copy()
                cmd.extend([
                '--heads',
                str(h),
                '--K',
                str(k)
                ])
                if args.model == 'aggr':
                    for e in parameters['epsilon']:
                        cmd.extend(['--epsilon', str(e)])
                        subprocess.run(cmd)
                else:
                    cmd.extend(['--epsilon', '0'])
                    subprocess.run(cmd)
    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Sweep over Parameters--')
    main(args)
