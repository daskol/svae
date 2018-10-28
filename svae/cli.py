#   encoding: utf8
#   filename: cli.py
"""Entry point for starting training of vanila VAE model. It is based on top of
VAE estimator which impliments sci-kit learn BaseEstimator interface. CLI fits
VAE from scratch. It prints objective value on standard output during learning
as well as depicts samples of latent space and reconstruction samples to
pictures on filesystem.
"""

from argparse import ArgumentParser
from os.path import abspath


parser = ArgumentParser(
    description=__doc__,
    epilog='Daniel Bershatsky <daniel.bershatsky@skolkovotech.ru, 2018')

parser.add_argument('--batch-size', metavar='POINTS',
                    nargs=1, default=[100], type=int,
                    help='Batch size.')

parser.add_argument('--datadir', metavar='DIR',
                    nargs=1, default=['data'], type=str,
                    help='Dataset directory.')

parser.add_argument('--noepochs', metavar='EPOCHS',
                    nargs=1, default=[15], type=int,
                    help='Number of training epochs.')

parser.add_argument('--nohiddens', metavar='UNITS',
                    nargs=1, default=[400], type=int,
                    help='Number of units in hidden layers.')

parser.add_argument('--nolatents', metavar='UNITS',
                    nargs=1, default=[20], type=int,
                    help='Size of latent space.')

parser.add_argument('--nosamples', metavar='SAMPLES',
                    nargs=1, default=[1], type=int,
                    help='Number of samples generated on decoding.')

parser.add_argument('--outdir', metavar='DIR',
                    nargs=1, default=['output'], type=str,
                    help='Output directory.')

parser.add_argument('--show-every', metavar='ITERS',
                    nargs=1, default=[100], type=int,
                    help='How often print learning status.')

parser.add_argument('--seed', metavar='SEED',
                    nargs=1, default=[42], type=int,
                    help='Random seed for pseudo RNG.')

parser.add_argument('dataset',
                    nargs=1, choices=('frey-face', 'mnist'),
                    help='Dataset to use')

parser.add_argument('decoder',
                    nargs='?', default='bernoulli',
                    choices=('bernoulli', 'gaussian'),
                    help='Type of probabilistic decoder.')

args = parser.parse_args()


def main():
    # Put imports here in order to speed up getting help.
    from torch import Tensor, float32
    from torchvision.datasets.mnist import MNIST
    from .datasets import FreyFace
    from .estimators import VAE

    # All bussiness starts here.
    if args.dataset[0] == 'frey-face':
        ff = FreyFace(args.datadir[0], download=True)
        dataset = ff.data.type(float32) / 255
    elif args.dataset[0] == 'mnist':
        mnist = MNIST(args.datadir[0], download=True)
        dataset = mnist.train_data.reshape(-1, 28 * 28)  # type: Tensor
        dataset = dataset.type(float32) / 255
    else:
        raise ValueError('Unknown dataset identifier was given.')

    print('Estimator:         ', 'vanila vae')
    print('Decoder:           ', args.decoder)
    print('Dataset:           ', args.dataset[0])
    print('Num of epochs:     ', args.noepochs[0])
    print('Batch size:        ', args.batch_size[0])
    print('Output directory:  ', abspath(args.outdir[0]))
    print()

    vae = VAE(nohiddens=args.nohiddens[0],
              nolatents=args.nolatents[0],
              nosamples=args.nosamples[0],
              noepochs=args.noepochs[0],
              batch_size=args.batch_size[0],
              show_every=args.show_every[0],
              decoder=args.decoder,
              outdir=args.outdir[0])
    vae.fit(dataset)
