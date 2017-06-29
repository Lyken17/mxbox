import logging
import multiprocessing as mp
import Queue
import atexit
import random
from multiprocessing import Pool

import mxnet as mx
import numpy as np
# FIXME: use PIL backend instead of opencv to read image
# import cv2

from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.random.permutation(len(self.data_source)).long())

    def __len__(self):
        return len(self.data_source)


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError("__getitem__() shoud be overided")

    def __len__(self):
        raise NotImplementedError("__len__() shoud be overided")

    @staticmethod
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

class TestDataset(Dataset):
    def __init__(self, root="../../data", transform=None, target_transform=None, batch_size=32):
        super(TestDataset, self).__init__()

        self.root = root

        self.batch_size = 32
        self.data_shapes = [mx.io.DataDesc(name='data', shape=(32, 13, 128, 128), layout='NCHW')]
        self.label_shapes = [mx.io.DataDesc(name='softmax_label', shape=(32,), layout='N')]

        self.transform = transform
        self.label_transform = target_transform

        with open(osp.join(root, 'caltech-256-60-train.lst'), 'r') as fp:
            self.flist = [line.strip().split() for line in fp.readlines()]

    def __getitem__(self, index):
        label, id, path, = self.flist[index]
        data = self.pil_loader(osp.join(self.root, '256_ObjectCategories', path))

        if self.transform is not None:
            data = self.transform(data)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return data, label

    def __len__(self):
        return len(self.flist)


import transformer
import DataLoader

if __name__ == "__main__":
    preprocess = transformer.Compose([
        transformer.Scale(512),
        transformer.RandomHorizontalFlip(),
        transformer.RandomCrop(256),
    ])

    loader = TestDataset(root='../../data', transform=preprocess)

    for _ in range(10):
        print(loader[0][0].size)
        plt.imshow(loader[0][0])
        plt.show()
