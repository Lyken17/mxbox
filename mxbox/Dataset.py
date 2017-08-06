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
