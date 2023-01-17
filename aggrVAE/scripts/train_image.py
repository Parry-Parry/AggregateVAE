from argparse import ArgumentParser
import os
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch
from aggrVAE.util.datamodule import ImageDataModule
from aggrVAE.image_model.vae import classifier_head, VAEclassifier
from aggrVAE.image_model.ensemble_vae import ensembleVAEclassifier


parser = ArgumentParser()

parser.add_argument('-data_source', type=str)
parser.add_argument('-data_sink', type=str)
parser.add_argument('-log_dir', type=str)
parser.add_argument('-ds', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--seed', type=int, default=8008)
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--latent', type=int, default=20)
parser.add_argument("--accum", default=1)

CONV = [32, 32, 64, 64]
LINEAR = [512, 256, 128]

def main(args):

    assert os.path.exists(args.log_dir)
    root_path = os.path.join(args.log_dir, f"{args.ds}.{args.K}.{args.heads}")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        chkpt = os.path.join(root_path, 'checkpoints')
        os.mkdir(chkpt)
        history = os.path.join(root_path, 'history')
        os.mkdir(history)

    logger = CSVLogger(root_path, name=f"{args.ds}.{args.K}.{args.heads}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(root_path, 'checkpoints'), save_weights_only=True, every_n_epochs=10)

    data_name = f'{args.ds}{args.K}{args.seed}.pkl'
    data = ImageDataModule(os.path.join(args.data_source, data_name), args.ds, args.batch_size, args.data_sink)

    if args.heads > 1:
        heads = [classifier_head(CONV, LINEAR, data.height, data.channels, data.classes) for i in range(args.heads)]
        model = ensembleVAEclassifier(heads, args.num_heads, 512, args.latent, data.classes, data.height, data.channels)
    else:
        head = classifier_head(CONV, LINEAR, data.height, data.channels, data.classes)
        model = VAEclassifier(head, 512, args.latent, data.classes, data.height, data.channels)

    trainer = pl.Trainer(max_epochs=args.epochs, logger=logger, accelerator='auto', devices=1 if torch.cuda.is_available() else None, accumulate_grad_batches=args.accum, callbacks=[checkpoint_callback])
    trainer.fit(model, data)
    hist = trainer.test(data)

    with open(os.path.join(root_path, 'history', f'history.{args.ds}.{args.K}.{args.heads}.pkl'), 'wb') as f:
        pickle.dump(f, hist)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)