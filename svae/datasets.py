#   encoding: utf8
#   filename: dataset.py

from os import makedirs
from os.path import exists, join

from PIL.Image import Image, fromarray
from scipy.io import loadmat

from torch import Tensor, tensor
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class FreyFace(Dataset):
    """This class implements pytorch Dataset interface. It could download raw
    data from origin WEB site or simply load images from filesystem.

    :param rootdir: Directory where to look for dataset or download it to.
    :param download: Download dataset or not.
    """

    __slots__ = ('rootdir', 'data', 'datalen')

    FILENAME = 'frey-face.mat'
    MD5 = '5aeb024d42b8a6091f30e5eb18c8a48d'
    URL = 'https://cs.nyu.edu/~roweis/data/frey_rawface.mat'

    def __init__(self, rootdir: str, download: bool=False):
        self.rootdir = rootdir

        if download:
            self.download(self.rootdir)

        self.data = self.read(self.rootdir)  # type: Tensor
        self.datalen = self.data.shape[0]

    def __getitem__(self, index: int) -> Image:
        return fromarray(self.data[:, :, index].numpy(), mode='L')

    def __len__(self) -> int:
        return self.datalen

    def download(self, root: str):
        if not exists(join(root, FreyFace.FILENAME)):
            makedirs(root, exist_ok=True)
            download_url(FreyFace.URL, root, FreyFace.FILENAME, FreyFace.MD5)

    def read(self, root: str) -> Tensor:
        mat = loadmat(join(root, FreyFace.FILENAME))
        images = mat['ff'].T
        return tensor(images)
