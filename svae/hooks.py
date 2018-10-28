#   encoding: utf8
#   filename: hooks.py

from os.path import join

from matplotlib.pyplot import close
from numpy.random import randn
from sklearn.base import BaseEstimator
from torch import Tensor

from .history import History
from .visual import visualize_latents, visualize_loss, visualize_reconstruction


class Hook:
    """Hook represents object that have three handlers which could be called
    before training loop, after training loop and during training. This class
    should be extended in subclasses.
    """

    def hook(self, *args, **kwargs):
        pass

    def prehook(self, *args, **kwargs):
        pass

    def posthook(self, *args, **kwargs):
        pass


class CombinedHook(Hook):
    """CombinedHook combines multiple hooks in list and calls them sequentially
    when this hook is called. So, it just forwards calls to underline hooks.

    :param hooks: Underline list of hooks.
    """

    def __init__(self, *hooks):
        self.hooks = list(hooks)

    def add(self, hook, *args, **kwargs):
        """Function constructs hook with a given arguments and appends newly
        created object to hook list.

        :param hook: Hook type or any callable object that construct hook.
        :return: Return this hook. It is usefull for chaining.
        """
        self.hooks.append(hook(*args, **kwargs))
        return self

    def hook(self, *args, **kwargs):
        for hook in self.hooks:
            hook.hook(*args, **kwargs)

    def prehook(self, *args, **kwargs):
        for hook in self.hooks:
            hook.prehook(*args, **kwargs)

    def posthook(self, *args, **kwargs):
        for hook in self.hooks:
            hook.posthook(*args, **kwargs)


class LatentSamplerHook(Hook):
    """LatentSamplerHook samples random points from latent space then estimates
    datapoints in real domain and visualizes them.

    :param outdir: Output directory for pictures.
    """

    def __init__(self, nolatents: int, edge: int=10, outdir: str='output/'):
        self.latents = Tensor(randn(edge * edge, nolatents))
        self.edge = edge
        self.outdir = outdir

    def hook(self, model, history: History):
        last = history.last()
        filename = 'latent-space-%02de-%03db.png' % (last[1], last[2])

        X = model.inverse_transform(self.latents)

        img = visualize_latents(X, self.edge)
        img.save(join(self.outdir, filename))

    def posthook(self, model, history: History):
        return self.hook(model, history)


class LossHook(Hook):
    """LossHook prints loss information on standard output during training. At
    the end of training it plots chart how loss and its components evolve.

    :param outdir: Output directory for pictures.
    """

    def __init__(self, outdir: str='output/'):
        self.outdir = outdir
        self.filename = join(self.outdir, 'loss.png')

    def hook(self, model: BaseEstimator, history: History):
        last = history.last()
        part = ['[\033[1;30m%05d/%02d:%03d\033[0m]',
                '\033[34mKL\033[0m = %8.3f   ',
                '\033[35mRE\033[0m = %8.3f   ',
                '\033[36mELBO\033[0m = %8.3f']
        line = ' '.join(part)
        print(line % last)

    def posthook(self, model: BaseEstimator, history: History):
        last = history.last()
        segs = ['              ',
                '\033[1;34mKL\033[0;1m = %8.3f   ',
                '\033[1;35mRE\033[0;1m = %8.3f   ',
                '\033[1;36mELBO\033[0;1m = %8.3f',
                '\033[0m']
        line = ' '.join(segs)
        print(line % last[3:])

        fig = visualize_loss(history)
        fig.savefig(self.filename)
        close(fig)


class RekonstruktHook(Hook):
    """RekonstruktHook apply VAE on given datapoints and then collate origin
    and reconstructed images in one picture.

    :param datapoints: Original datapoints that should be reconstructed.
    :param outdir: Output directory for pictures.
    """

    def __init__(self, datapoints: Tensor, outdir: str='output/'):
        self.datapoints = datapoints
        self.outdir = outdir

    def hook(self, model: BaseEstimator, history: History):
        noimages = self.datapoints.shape[0]
        latvar = model.transform(self.datapoints)
        konstrukt = model.inverse_transform(latvar())

        last = history.last()
        filename = 'rekonstrukt-%02de-%03db.png' % (last[1], last[2])

        img = visualize_reconstruction(self.datapoints, konstrukt, noimages)
        img.save(join(self.outdir, filename))

    def posthook(self, model, history: History):
        self.hook(model, history)
