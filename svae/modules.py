#   encoding: utf8
#   filename: modules.py

import torch as T
import torch.nn as N
import torch.nn.init as I
import torch.nn.functional as F

from typing import Tuple
from numpy import log, pi
from torch import Tensor, tensor


class VAEBase(N.Module):
    """VAEBase is a base module for different VAE implemention. The difference
    appear in implimentation of probabilistic decoder. This results in
    different approaches in estimation of reconstruction errors. In the same
    time there is a bunch of method that stay unchanged.

    :param noinputs: Number of input features.
    :param nohiddens: Number of units in hidden layer.
    :param nolatents: Dimension of latent space.
    :param nosamples: Number of samples generated on decodeing stage.
    """

    def __init__(self, noinputs: int, nohiddens: int, nolatents: int,
                 nosamples: int, **kwargs):

        super().__init__()

        self.noinputs = noinputs
        self.nohiddens = nohiddens
        self.nolatents = nolatents
        self.nosamples = nosamples

        # See Appendix C on page 11.
        self.fc1 = N.Linear(noinputs, nohiddens)
        self.fc2 = N.Linear(nohiddens, nolatents)
        self.fc3 = N.Linear(nohiddens, nolatents)

        # See page 7.
        I.normal_(self.fc1.weight, std=0.01)
        I.normal_(self.fc2.weight, std=0.01)
        I.normal_(self.fc3.weight, std=0.01)

        # See page 7.
        I.normal_(self.fc1.bias, std=0.01)
        I.normal_(self.fc2.bias, std=0.01)
        I.normal_(self.fc3.bias, std=0.01)

    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError('Actual implementation of decoder must be '
                                  'done in child classes.')

    def encode(self, x: Tensor) -> Tensor:
        assert self.noinputs == x.shape[-1], 'Wrong number of inputs.'

        # See Appendix C.2 on page 11.
        hiddens = T.tanh(self.fc1(x))
        mu = self.fc2(hiddens)
        logsigma2 = self.fc3(hiddens)  # \log \sigma^2
        return mu, logsigma2

    def sample(self, mu: Tensor, logsigma2: Tensor) -> Tensor:
        assert mu.shape == logsigma2.shape, 'Mismatch of parameters shape.'
        assert mu.shape[-1] == self.nolatents, 'Wrong size of latent space.'

        std = logsigma2.mul(0.5).exp()
        eps = T.randn_like(mu)
        z = mu + std * eps
        return z

    def kl(self, mu: Tensor, logsigma2: Tensor) -> Tensor:
        """This function returns estimated value of KL-divergence in case of
        normal Gaussian prior. See Appendix B on pages 10-11.

        :param mu: Mean of posterior probability.
        :param logsigma2: Natural logarithm of squared of posterior variance.
        :return: KL-divergence estimated per batch.
        """
        return 0.5 * T.sum(mu ** 2 + logsigma2.exp() - logsigma2 - 1)

    def re(self, *args):
        raise NotImplementedError('Reconstruction error estimation is depends '
                                  'on model of decoder network and must be '
                                  'defined in subclasses.')


class VAEBernoulliDecoder(VAEBase):

    def __init__(self, noinputs: int, nohiddens: int=400, nolatents: int=20,
                 nosamples: int=1, **kwargs):

        super().__init__(noinputs, nohiddens, nolatents, nosamples)

        # See Appendix C.1 on page 11.
        self.fc4 = N.Linear(nolatents, nohiddens)
        self.fc5 = N.Linear(nohiddens, noinputs)

        # See page 7.
        I.normal_(self.fc4.weight, std=0.01)
        I.normal_(self.fc5.weight, std=0.01)
        I.normal_(self.fc4.bias, std=0.01)
        I.normal_(self.fc5.bias, std=0.01)

    def decode(self, z: Tensor) -> Tensor:
        assert self.nolatents == z.shape[-1], 'Wrong size of latent space.'

        # See Appendix C.1 on page 11.
        hiddens = T.tanh(self.fc4(z))
        output = T.sigmoid(self.fc5(hiddens))
        return output

    def re(self, x: Tensor, X: Tensor) -> Tensor:
        """Estimate reconstruction error of VAE model for batch of real
        datapoints. See Section 3 on page 5.

        :param x: Real datapoints.
        :param X: Samples in real domain generated from latent space.
        :return: Estimated value of reconstruction error.
        """
        assert x.shape == X.shape, 'Wrong shape on datapoints and samples.'
        return F.binary_cross_entropy(X, x, reduction='sum')


class VAEGaussianDecoder(VAEBase):

    def __init__(self, noinputs: int, nohiddens: int=400, nolatents: int=20,
                 nosamples: int=1, **kwargs):

        super().__init__(noinputs, nohiddens, nolatents, nosamples)

        # See Appendix C.2 on page 11.
        self.fc4 = N.Linear(nolatents, nohiddens)
        self.fc5 = N.Linear(nohiddens, noinputs)
        self.fc6 = N.Linear(nohiddens, noinputs)

        # See page 7.
        I.normal_(self.fc4.weight, std=0.01)
        I.normal_(self.fc5.weight, std=0.01)
        I.normal_(self.fc6.weight, std=0.01)
        I.normal_(self.fc4.bias, std=0.01)
        I.normal_(self.fc5.bias, std=0.01)
        I.normal_(self.fc6.bias, std=0.01)

    def decode(self, z: Tensor) -> Tensor:
        assert self.nolatents == z.shape[-1], 'Wrong size of latent space.'

        # See Appendix C.2 on page 11.
        hiddens = T.tanh(self.fc4(z))
        mu = T.sigmoid(self.fc5(hiddens))
        logsigma2 = self.fc6(hiddens)
        return mu, logsigma2

    def re(self, x: Tensor, X: Tuple[Tensor, Tensor]) -> Tensor:
        assert len(X) == 2, 'Wrong number of params of X distributiom.'

        mu, logsigma2 = X

        assert mu.shape == logsigma2.shape, 'Wrong shape of parameters.'
        assert x.shape[0] == mu.shape[0], 'Batch size mismatch.'
        offset = tensor(log(2 * pi))
        re = 0.5 * T.sum(offset + logsigma2 + (x - mu) ** 2 / logsigma2.exp())
        return re
