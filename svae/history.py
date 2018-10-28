#   encoding: utf8
#   filename: history.py

from collections import namedtuple
from typing import Tuple


class History(namedtuple('History', ['epoch', 'batch', 'kl', 're', 'elbo'])):
    """History is container for storing information about loss and training
    process.
    """

    def __init__(self, *args, **kwargs):
        self.index = 0

    def append(self, epoch: int, batch: int, kl: float, re: float):
        """Append appends another one datapoint to collected statistics.

        :param epoch: Current epoch index of training.
        :param batch: Current batch index of training.
        :param kl: Value of KL-divergence estimated per sample.
        :param re: Value of reconstruction error estimated per sample.
        """
        self.epoch[self.index] = epoch
        self.batch[self.index] = batch
        self.kl[self.index] = kl
        self.re[self.index] = re
        self.elbo[self.index] = -(kl + re)
        self.index += 1

    def last(self) -> Tuple[int, int, int, float, float, float]:
        """Last gives last appended datapoint.

        :return: Tuple of (ordinal number, epoch, batch, KL, RE, ELBO).
        """
        if self.index == 0:
            raise ValueError('There is nothing in history.')
        else:
            last = self.index - 1
            return (last, self.epoch[last], self.batch[last],
                    self.kl[last], self.re[last], self.elbo[last])
