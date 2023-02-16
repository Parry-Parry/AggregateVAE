from argparse import ArgumentParser
import logging
import os
import pickle

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pl_bolts.models.autoencoders.components import resnet18_encoder

import torch
from torch import nn

from aggrVAE.util.datamodule import MNISTDataModule, CIFAR10DataModule
from aggrVAE.image.classifier.seq import classifier_head, Classifier
from aggrVAE.image.classifier.ensemble import EnsembleClassifier


parser = ArgumentParser()

parser.add_argument('-data_sink', type=str)
parser.add_argument('-log_dir', type=str)
parser.add_argument('-ds', type=str)
parser.add_argument('--data_source', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--seed', type=int, default=8008)
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--latent', type=int, default=20)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument("--accum", default=1)

parser.add_argument('--verbose', action='store_true')

LINEAR = [512, 256, 128]

datasets = {
    'MNIST' : MNISTDataModule,
    'CIFAR10' : CIFAR10DataModule
}

def main(args):
    assert os.path.exists(args.log_dir)
    root_path = os.path.join(args.log_dir, f"{args.ds}.{args.heads}")
    if not os.path.exists(root_path):
        logging.info('ROOT does not exist, creating...')
        os.mkdir(root_path)
        chkpt = os.path.join(root_path, 'checkpoints')
        os.mkdir(chkpt)
        history = os.path.join(root_path, 'history')
        os.mkdir(history)
        logging.info(f'ROOT made at {root_path}')

    logger = CSVLogger(root_path, name=f"{args.ds}.{args.heads}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(root_path, 'checkpoints'), save_weights_only=True, save_top_k=1, every_n_epochs=5)

    assert args.ds in datasets.keys(), f"No Aggregates Present for {args.ds}"
    logging.info('Initialising DataModule...')
    data = datasets[args.ds](args.batch_size, args.num_workers, args.data_sink)

    if args.heads > 1:
        model = EnsembleClassifier(LINEAR, args.heads, args.epsilon, args.latent, data.classes, data.channels)
    else:
        model = Classifier(LINEAR, args.epsilon, args.latent, data.classes, data.channels)

    logging.info(f'Training for {args.epochs} epochs...')
    trainer = pl.Trainer(default_root_dir=os.path.join(root_path, 'checkpoints'), max_epochs=args.epochs, logger=logger, accelerator='auto', devices=1 if torch.cuda.is_available() else None, accumulate_grad_batches=args.accum, callbacks=[checkpoint_callback])
    trainer.fit(model, data)
    
    trainer.test(model=model, dataloaders=data.test_dataloader())
    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Initialising IPAT Image Classifier Training--')
    main(args)
