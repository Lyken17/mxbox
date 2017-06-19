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

    def __init__(self):
        """
            set all required variables ready, see implementation below for more details 
        """
        super(DataLoader, self).__init__()

        # shape related variables
        self.data_shape = (32, 3, 128, 12)
        self.label_shape = (32, 1)
        self.batch_size = self.data_shape[0]
        self.image_size = self.data_shape[1:]

        # list related variables
        self.loader_list = []  # TODO: add a function interface
        self.current = 0
        self.total = len(self.loader_list)
        self.random_shuffle = False

        # multi thread acceleration
        self.read_threads = 1

        # transformation
        self.transform = None

        raise NotImplementedError('you must override __init__() you self')

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
            return self.load_batch()

        raise NotImplementedError('you must override next() you self')

    def load_batch(self):
        # make it static, unreachable from outside
        index_list = range(self.current, self.current + self.batch_size)

        if self.read_threads > 1:
            p = Pool(self.read_threads)  # TODO: add pin memory to optimize speed
            batch = p.map(self.get_single_pair, index_list)
        else:
            batch = []
            for ind in index_list:
                batch.append(self.get_single_pair[ind])
        return batch

    def get_single_pair(self, index):
        data = self.get_single_data(index)
        label = self.get_single_label(index)
        return [data, label]

    def get_single_data(self, index):
        raise NotImplemented

    def get_single_label(self, index):
        raise NotImplemented

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
        self.current = 0
        if self.random_shuffle:
            random.shuffle(self.loader_list)



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
