from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import torch
from aggrVAE.util.datamodule import ImageDataModule
from aggrVAE.image_model.vae import classifier_head, VAEclassifier
from aggrVAE.image_model.ensemble_vae import ensembleVAEclassifier


parser = ArgumentParser()

parser.add_argument('-data_source', type=str)
parser.add_argument('-data_sink', type=str)
parser.add_argument('-ds', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n_class', type=int, default=10)
parser.add_argument('--in_height', type=int, default=32)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--seed', type=int, default=8008)
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--latent', type=int, default=20)
parser.add_argument("--accum", default=1)

CONV = []
LINEAR = []

def main(args):
    data_name = f'{args.ds}{args.K}{args.seed}.pkl'
    data = ImageDataModule(os.path.join(args.data_source, data_name), args.ds, args.batch_size, args.data_sink)

    if args.heads > 1:
        heads = [classifier_head(CONV, LINEAR, data.height, data.channels, data.classes) for i in range(args.heads)]
        model = ensembleVAEclassifier(heads, args.num_heads, 512, args.latent, data.classes, data.height, data.channels)
    else:
        head = classifier_head(CONV, LINEAR, data.height, data.channels, data.classes)
        model = VAEclassifier(head, 512, args.latent, data.classes, data.height, data.channels)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', devices=1 if torch.cuda.is_available() else None, accumulate_grad_batches=args.accum)
    trainer.fit(model, data)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)