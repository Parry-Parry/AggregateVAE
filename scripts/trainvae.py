from argparse import ArgumentParser
import pytorch_lightning as pl


def main(args):
    model = LightningModule()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)