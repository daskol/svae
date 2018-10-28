#   encoding: utf8
#   filename: visual.py

from PIL.Image import Image, fromarray
from matplotlib.pyplot import Figure, figure
from numpy import arange, swapaxes, uint8, zeros
from torch import Tensor

from .history import History


def visualize_latents(X: Tensor, edge: int) -> Image:
    """Visualize sampled points from latent space. It forms image square
    lattice of size edge x edge.

    :param X: Datapoints sampled from latent space.
    :param edge: Number of images along both X and Y axis.
    :return: Image object with painted datapoints.
    """
    # Compute height of a single image. Width of image in FreyFace or MNIST
    # dataset is known.
    height = 28
    width =  X.shape[1] // height

    # Fill canvas with sampled images.
    X = X.detach().numpy()
    X = X.reshape(-1, height, width)
    img = zeros((height * edge, width * edge))

    for i in range(edge):
        for j in range(edge):
            index = i + edge * j
            slice_y = slice(height * i, height * (i + 1))
            slice_x = slice(width * j, width * (j + 1))
            img[slice_y, slice_x] = X[index, :, :]

    return fromarray(uint8(img * 255))


def visualize_loss(history: History) -> Figure:
    """Function plots how ELBO estimation evolves with time.

    :param history: History object that aggregates statistics in runtime.
    :return: Figure object.
    """
    nopoints = arange(history.index)
    fig = figure(figsize=(16, 7))  # type: Figure
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nopoints, -history.kl, label='KL')
    ax.plot(nopoints, -history.re, label='RE')
    ax.plot(nopoints, history.elbo, label='ELBO')
    ax.set_xlabel('num of batches')
    ax.set_ylabel('ELBO')
    ax.grid(True)
    ax.legend()
    return fig


def visualize_reconstruction(x: Tensor, X: Tensor, noimages: int=10) -> Image:
    """The function takes several first images, plot them, and save to
    png-file.

    :param x: Original datapoint.
    :param X: Reconstructed datapoint.
    :param noimages: Number of images to plot.
    :return: Image object.
    """
    height = 28 # Both FreyFace and MNIST datasets has the same width.
    width = x.shape[1] // height

    noimages = min(noimages, x.shape[0])  # Adjust to batch size.

    # Preprocess original images.
    x = x[:noimages, :].detach().numpy()  # type: ndarray
    x = x.reshape(noimages, height, width)
    x = swapaxes(x, 0, 1)

    # Preprocess reconstruction images.
    X = X[:noimages, :].detach().numpy()  # type: ndarray
    X = X.reshape(noimages, height, width)
    X = swapaxes(X, 0, 1)

    # Collate small images in a single canvas.
    img = zeros((2 * height, noimages * width))  # type: ndarray
    img[:height, :] = x.reshape(height, -1)
    img[height:, :] = X.reshape(height, -1)

    return fromarray(uint8(img * 255))
