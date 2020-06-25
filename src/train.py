from argparse import ArgumentParser
from .model import TwoTower
from pytorch_lightning import Trainer

def main(args):
    trainer = Trainer.from_argparse_args(args)
    dict_args = vars(args)
    model = TwoTower(**dict_args)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = TwoTower.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
