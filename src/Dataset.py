import mxnet as mx
import logging
import multiprocessing as mp
import numpy as np
import Queue
import atexit
import random
from multiprocessing import Pool

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


class ImageFolder(mx.io.DataIter):
    '''
    To read  

    Example:
        root/dog/xx01.jpg
        root/dog/xx02.jpg

        root/cat/xx01.jpg
        root/cat/xx02.jpg
    '''

    def __init__(self):
        pass


class SampleLoader(mx.io.DataIter):
    """
        In mxnet, 5 functions below is necessary for implementing a DataLoader
    """

    def __init__(self):
        """
            set all required variables ready, see implementation below for more details 
        """
        raise NotImplementedError('you must override __init__() you self')

    def next(self):
        """
        :return:
            mx.io.DataBatch(data = [data], label = [label]) 
                detailed explaination later        
        :raises:
            StopIteration: 
                if data loader reachs epoch end
        """
        raise NotImplementedError('you must override next() you self')

    @property
    def provide_data(self):
        """
        :return:
            [mx.io.DataDesc(), ... ]
                A list of mx.io.DataDesc, which describes all data input blocks in network  
        """
        raise NotImplementedError('you must override provide_data() you self')

    @property
    def provide_label(self):
        """
        :return:
            [mx.io.DataDesc(), ... ]
                A list of mx.io.DataDesc, which describes all label input blocks in network  
        """
        raise NotImplementedError('you must override provide_label() you self')

    def reset(self):
        """
        reset variables related to iterations, such as current_index, shuffle, etc
        """
        raise NotImplementedError('you must override reset() you self')


class DataLoader(mx.io.DataIter):
    """
        In mxnet, 5 functions below is necessary for implementing a DataLoader
    """

    def __init__(self, read_threads=1):
        """
            set all required variables ready, see implementation below for more details 
        """
        super(DataLoader, self).__init__()

        # shape related variables
        self.data_shapes = [mx.io.DataDesc(name='data', shape=(32, 3, 128, 128), layout='NCHW')]
        self.label_shapes = [mx.io.DataDesc(name='softmax_label', shape=(32,), layout='N')]
        self.batch_size = 32
        # self.image_size = self.data_shape[1:]

        # list related variables
        self.root = None

        self.loader_list = []  # TODO: add a function interface
        self.current = 0
        self.total = len(self.loader_list)
        self.random_shuffle = False

        # multi thread acceleration
        self.read_threads = 1

        # transformation
        self.transform = None

        # raise NotImplementedError('you must override __init__() you self')

    def next(self):
        """
        :return:
            mx.io.DataBatch(data = [data], label = [label]) 
                detailed explanation later        
        :raises:
            StopIteration: 
                if data loader reaches epoch end
        """
        if self.current + self.batch_size > self.total:
            raise StopIteration  # reach epoch end
        else:
            return self.get_batch()

            # TODO: remove this comment when construction finishes
            # raise NotImplementedError('you must override next() you self')

    def get_batch(self):
        # make it static, unreachable from outside
        index_list = range(self.current, self.current + self.batch_size)

        if self.read_threads > 1:
            p = Pool(self.read_threads)  # TODO: add pin memory to optimize speed
            batch = p.map(self.__getitem__, index_list)
        else:
            batch = []
            for ind in index_list:
                batch.append(self.get_single_pair(ind))

        # TODO: make batch here

        return batch

    def get_single_pair(self, index):
        # TODO: will be removed. Replace it by __getitem__
        return self.__getitem__(index)

    def __getitem__(self, index):
        """
        :param index(int): Index
        :return: 
            tuple: (data, label) where data and label are collections 
        """
        raise NotImplementedError('you must override __getitem__()')

    def get_single_data(self, index):
        raise NotImplemented

    def get_single_label(self, index):
        raise NotImplemented

    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    @property
    def provide_data(self):
        """
        :return:
            [mx.io.DataDesc(), ... ]
                A list of mx.io.DataDesc, which describes all data input blocks in network  
        """
        return self.data_shapes
        # raise NotImplementedError('you must override provide_data() ')

    @property
    def provide_label(self):
        """
        :return:
            [mx.io.DataDesc(), ... ]
                A list of mx.io.DataDesc, which describes all label input blocks in network  
        """
        return self.label_shapes
        # raise NotImplementedError('you must override provide_label() ')

    def reset(self):
        """
        reset variables related to iterations, such as current_index, shuffle, etc
        """
        self.current = 0
        if self.random_shuffle:
            random.shuffle(self.loader_list)

    def __len__(self):
        return len(self.loader_list)


class TestLoader(DataLoader):
    def __init__(self, root="../data", transform=None, target_transform=None, batch_size=32):
        super(TestLoader, self).__init__()

        self.batch_size = 32
        self.data_shapes = [mx.io.DataDesc(name='data', shape=(32, 13, 128, 128), layout='NCHW')]
        self.label_shapes = [mx.io.DataDesc(name='softmax_label', shape=(32,), layout='N')]

        self.transform = None
        self.label_transform = None
        self.flist = []

    def __getitem__(self, index):
        path, label = self.flist[index]
        data = self.pil_loader(path)

        if self.transform is not None:
            data = self.transform(data)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return data, label

    def __len__(self):
        return len(self.loader_list)


if __name__ == "__main__":
    loader = TestLoader()
    print(loader.provide_data)
