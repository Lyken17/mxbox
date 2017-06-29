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
from collections import namedtuple

if __name__ == "__main__":
    preprocess = transformer.Compose([
        transformer.Scale(512),
        transformer.RandomHorizontalFlip(),
        transformer.RandomCrop(256),
        transformer.PILToNumpy(),
        transformer.Lambda(lambd=lambda img: np.swapaxes(img, 0, 2)),
        transformer.Lambda(lambd=lambda img: img.reshape(1, img.shape[0], img.shape[1],  img.shape[2]))  # 1xCxWxH)
    ])

    feedin_shapes = {
        'batch_size' : 32,
        'data' : [mx.io.DataDesc(name='data', shape=(32, 3, 128, 128), layout='NCHW')],
        'label': [mx.io.DataDesc(name='softmax_label', shape=(32, ), layout='N')]
    }

    dst = TestDataset(root='../../data', transform=preprocess)

    for _ in range(10):
        print(dst[0][0].shape)
