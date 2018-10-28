#   encoding: utf8
#   filename: estimators.py

from logging import getLogger
from math import ceil
from os import makedirs
from os.path import join
from typing import Tuple

from numpy import arange, ndarray, rollaxis, swapaxes, zeros
from sklearn.base import BaseEstimator, TransformerMixin

from torch import Tensor
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from .history import History
from .hooks import CombinedHook, LatentSamplerHook, LossHook, RekonstruktHook
from .modules import VAEBase, VAEBernoulliDecoder, VAEGaussianDecoder


class LatentVariable:
    """Class LatentVariable models probabilistic nature of latent variables. It
    allows to extend sklearn base mixing in way to handle distributions as
    result of real data transformation.

    :param VAEBase: Model implemented in PyTorch that is in VAE family.
    :param mu: Mean value of latent variables.
    :param logsigma2: Transformed covariance of latent variables.
    """

    def __init__(self, model: VAEBase, mu: Tensor, logsigma2: Tensor):
        self.model = model
        self.mu = mu
        self.logsigma2 = logsigma2

    def __call__(self) -> Tensor:
        """Sample points from latent space.
        """
        return self.model.sample(self.mu, self.logsigma2)


class VAE(BaseEstimator, TransformerMixin):
    """
    :param decoder: Address neural network in decoder. Possible values are
                    bernoulli and gaussian.
    """

    def __init__(self, nohiddens: int=400, nolatents: int=20, nosamples: int=1,
                 noepochs: int=15, batch_size: int=100, show_every: int=100,
                 decoder: str='bernoulli', outdir: str='output/'):

        super(BaseEstimator, self).__init__()

        makedirs(outdir, exist_ok=True)

        self.noinputs = None
        self.nohiddens = nohiddens
        self.nolatents = nolatents
        self.nosamples = nosamples
        self.noepochs = noepochs
        self.batch_size = batch_size
        self.show_every = show_every
        self.outdir = outdir

        if decoder == 'bernoulli':
            self.fit = self.__fit_bernoulli
        elif decoder == 'gaussian':
            self.fit = self.__fit_gaussian
        else:
            raise ValueError(f'Unknown decoder type: "{decoder}".')

        self.logger = getLogger(__name__)
        self.model = None
        self.opt = None

    def fit(self, X):
        """Method fit is overloaded during construction of VAE estimator. See
        constructor for details.
        """

    def transform(self, X: Tensor) -> LatentVariable:
        latvar = LatentVariable(self.model, *self.model.encode(X))
        return latvar

    def inverse_transform(self, X: Tensor) -> Tensor:
        origin = self.model.decode(X)

        if isinstance(origin, tuple):
            return origin[0]
        else:
            return origin

    def __fit(self, dataset: Tensor, model: VAEBase):
        it = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        dur = self.noepochs * ceil(len(dataset) / self.batch_size)
        history = History(zeros(dur), zeros(dur), zeros(dur),
                          zeros(dur), zeros(dur))

        hooks = CombinedHook()
        hooks.add(LossHook)
        hooks.add(RekonstruktHook, dataset[:10, :])
        hooks.add(LatentSamplerHook, self.nolatents)
        hooks.prehook(self, history)

        self.model = model
        self.noinputs = model.noinputs
        self.opt = Adagrad(self.model.parameters(), lr=0.01)  # See Section 5.

        for epoch in range(self.noepochs):
            for i, x in enumerate(it):
                self.opt.zero_grad()

                # Apply model in the following steps:
                # (a) encode datapoint into latent space;
                # (b) sample points from latent space;
                # (c) decode sampled points from latent space.
                mu, logsigma2 = self.model.encode(x)
                z = self.model.sample(mu, logsigma2)
                X = self.model.decode(z)

                # Estimate KL-divergence and reconstruction error (RE).
                kl = self.model.kl(mu, logsigma2)
                re = self.model.re(x, X)

                # Do error backpropagation.
                loss = kl + re
                loss.backward()
                self.opt.step()

                # Aggregation runtime statistics.
                history.append(epoch=epoch,
                               batch=i,
                               kl=float(kl / self.batch_size),
                               re=float(re / self.batch_size))

                if i % self.show_every == 0:
                    hooks.hook(self, history)

        # Print status before exit.
        hooks.posthook(self, history)

        # Return itself for calls chaining.
        return self

    def __fit_bernoulli(self, dataset: Tensor):
        params = self.get_params()
        params['noinputs'] = dataset.shape[1]
        model = VAEBernoulliDecoder(**params)
        return self.__fit(dataset, model)

    def __fit_gaussian(self, dataset: Tensor):
        params = self.get_params()
        params['noinputs'] = dataset.shape[1]
        model = VAEGaussianDecoder(**params)
        return self.__fit(dataset, model)
