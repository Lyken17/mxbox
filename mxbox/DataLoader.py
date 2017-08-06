import logging
import multiprocessing as mp
import numpy as np

import Queue
import threading

import atexit
import random
import sys

from multiprocessing import Pool
from PIL import Image
import mxnet as mx

import multiprocessing as multiprocessing

if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)


class SampleLoader(mx.io.DataIter):
    """
        In mxnet, 5 functions below is necessary for implementing a DataLoader
    """

    def __init__(self):
        """
            set all required variables ready, see implementation below for more details 
        """
        raise NotImplementedError('you must override __init__() yourself')

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
        raise NotImplementedError('you must override provide_data() yourself')

    @property
    def provide_label(self):
        """
        :return:
            [mx.io.DataDesc(), ... ]
                A list of mx.io.DataDesc, which describes all label input blocks in network  
        """
        raise NotImplementedError('you must override provide_label() yourself')

    def reset(self):
        """
        reset variables related to iterations, such as current_index, shuffle, etc
        """
        raise NotImplementedError('you must override reset() yourself')

def collate_fn(batch):
    pass

import transforms


class default_collate(object):
    def __init__(self, feedin_shape):
        self.feedin_shape = feedin_shape

    def __call__(self, batch):
        data = {}
        label = {}

        for dsc in self.feedin_shape['data']:
            data[dsc.name] = []
        for dsc in self.feedin_shape['label']:
            label[dsc.name] = []

        for name in data:
            for entry in batch:
                data[name].append(entry[name])
            data[name] = transforms.mx.stack(data[name], axis=0)

        for name in label:
            for entry in batch:
                label[name].append(entry[name])
            label[name] = transforms.mx.stack(label[name], axis=0)

        return mx.io.DataBatch(data.values(), label.values())

class DataLoader(mx.io.DataIter):
    def default_collate_fn(self, batch):
        data = {}
        label = {}

        for dsc in self.provide_data:
            data[dsc.name] = []
        for dsc in self.provide_label:
            label[dsc.name] = []

        for name in data:
            for entry in batch:
                data[name].append(entry[name])
            data[name] = transforms.mx.stack(data[name], axis=0)

        for name in label:
            for entry in batch:
                label[name].append(entry[name])
            label[name] = transforms.mx.stack(label[name], axis=0)

        return mx.io.DataBatch(data=data.values(), provide_data=self.provide_data, label=label.values())

    def __init__(self, dataset, feedin_shape, collate_fn=default_collate, threads=1, shuffle=False):
        super(DataLoader, self).__init__()

        self.dataset = dataset
        self.threads = threads
        self.collate_fn = collate_fn(feedin_shape)
        # self.collate_fn = self.default_collate_fn

        # shape related variables

        self.data_shapes = feedin_shape['data']
        self.label_shapes = feedin_shape['label']
        self.batch_size = feedin_shape['batch_size']

        # loader related variables
        self.current = 0
        self.total = len(self.dataset)
        self.shuflle = shuffle
        self.map_index = list(range(self.total))

        # prepare for loading
        self.get_batch = self.get_batch_single_thread
        if self.threads > 1:  # multi process read
            from multiprocessing.dummy import Pool as ThreadPool
            # self.pool = multiprocessing.Pool(self.threads)
            self.pool = ThreadPool(self.threads)
            self.get_batch = self.get_batch_multi_thread

        self.reset()

    def next(self):
        if self.current + self.batch_size > self.total:
            # reach end
            self.reset()
            raise StopIteration
        else:
            batch = self.get_batch()
            try:
                return self.collate_fn(batch)
            except AttributeError:
                print(batch)
                exit(-1)

    def get_single(self, index):
        # to ease
        idx = self.map_index[index]
        return self.dataset[idx]

    def get_batch_single_thread(self):
        entry = [None] * self.batch_size
        for idx in range(self.batch_size):
            entry[idx] = self.get_single(self.current + idx)
        self.current += self.batch_size
        return entry

    def get_batch_multi_thread(self):
        idx = range(self.current, self.current + self.batch_size)
        entry = self.pool.map(self.get_single, idx)
        self.current += self.batch_size
        return entry

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
        if self.shuflle:
            random.shuffle(self.map_index)
        return


class _DataLoader(mx.io.DataIter):
    """
        In mxnet, 5 functions below is necessary for implementing a DataLoader
    """

    def __init__(self, dataset, feedin_shape, read_threads=1, ):
        """
            set all required variables ready, see implementation below for more details 
        """
        super(_DataLoader, self).__init__()

        self.dataset = dataset

        ##################################################################################################
        # shape related variables
        # self.data_shapes = self.dataset.data_shapes
        # self.label_shapes = self.dataset.label_shapes
        # self.batch_size = self.dataset.batch_size

        self.data_shapes = feedin_shape['data']
        self.label_shapes = feedin_shape['label']
        self.batch_size = feedin_shape['batch_size']

        self.data_nums = len(self.provide_data)
        self.label_nums = len(self.provide_label)

        self.data_batch = [[None] * self.batch_size] * self.data_nums
        self.label_batch = [[None] * self.batch_size] * self.data_nums
        ##################################################################################################
        # loader related variables
        self.current = 0
        self.total = len(self.dataset)
        self.random_shuffle = False

        ##################################################################################################
        # multi thread acceleration
        self.read_threads = read_threads
        if self.read_threads > 1:
            # self.pool = Pool(self.read_threads)  # TODO: add pin memory to optimize speed
            self.producer = Queue.Queue()
            self.consumer = Queue.Queue()
            for _ in range(self.read_threads):
                t = threading.Thread(target=self.do_work, args=(self.producer, self.consumer))
                t.daemon = True
                t.start()

    def do_work(self, in_queue, out_queue):
        while True:
            index = in_queue.get()
            result = self.dataset[index]
            out_queue.put(result)
            in_queue.task_done()

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

    def load_batch(self):
        # make it static, unreachable from outside
        index_list = range(self.current, self.current + self.batch_size)

        if self.read_threads == 1:
            batch = []
            for ind in index_list:
                batch.append(self.__getitem__(ind))
        else:
            batch = []
            for ind in index_list:
                self.producer.put(ind)
            self.producer.join()
            for i in xrange(self.read_threads):
                batch.append(batch)
                # raise NotImplementedError
                # batch = self.pool.map(self.__getitem__, index_list)
        return batch

    def get_batch(self):
        # TODO: here is a wrong wrapping for collate
        batch = self.load_batch()
        #  [((data1, ..., dataN), (label1, ..., labelN)),
        #   ((data1, ..., dataN), (label1, ..., labelN)),
        #    ....
        #   ((data1, ..., dataN), (label1, ..., labelN))]

        # TODO: make batch here
        for ind in range(self.data_nums):
            self.data_batch[ind] = [batch[i][0][ind] for i in range(self.batch_size)]
        for ind in range(self.label_nums):
            self.label_batch[ind] = [batch[i][1][ind] for i in range(self.batch_size)]

        for ind in range(self.data_nums):
            # self.data_batch[ind] = np.concatenate(self.data_batch[ind], axis=0)
            # self.data_batch[ind] = mx.nd.array(self.data_batch[ind])
            self.data_batch[ind] = mx.nd.concatenate(self.data_batch[ind], axis=0)

        for ind in range(self.label_nums):
            # self.label_nums[ind] = np.concatenate(self.label_nums[ind], axis=0)
            # self.label_batch[ind] = mx.nd.array(self.label_batch[ind])
            self.label_batch[ind] = mx.nd.concatenate(self.label_batch[ind], axis=0)

        return mx.io.DataBatch(data=self.data_batch, label=self.label_batch)

    def __getitem__(self, index):
        """
        :param index(int): Index
        :return: 
            tuple: (data, label) where data and label are collections 
        """
        return self.dataset[index]

    @staticmethod
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
        # TODO: implement shuffle later
        # if self.random_shuffle:
        #     random.shuffle(self.loader_list)

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    return batch

'''
from torchloader import DataLoader as torchloader
from torchloader import DataLoaderIter as torchiter


class BoxLoader(mx.io.DataIter):
    """
        In mxnet, 5 functions below is necessary for implementing a DataLoader
    """

    def __init__(self, dataset, feedin_shape, num_workers=1, shuffle=False, collate_fn=collate_fn):
        """
            set all required variables ready, see implementation below for more details 
        """
        super(BoxLoader, self).__init__()

        self.dataset = dataset
        self.read_threads = num_workers
        self.collate_fn = collate_fn
        ##################################################################################################
        # shape related variables
        # self.data_shapes = self.dataset.data_shapes
        # self.label_shapes = self.dataset.label_shapes
        # self.batch_size = self.dataset.batch_size

        self.data_shapes = feedin_shape['data']
        self.label_shapes = feedin_shape['label']
        self.batch_size = feedin_shape['batch_size']

        self.data_nums = len(self.provide_data)
        self.label_nums = len(self.provide_label)

        self.data_batch = [[None] * self.batch_size] * self.data_nums
        self.label_batch = [[None] * self.batch_size] * self.data_nums
        ##################################################################################################
        # loader related variables
        self.current = 0
        self.total = len(self.dataset)
        self.random_shuffle = False

        ###############

        self.torchloader = torchloader(self.dataset, batch_size=self.batch_size,
                                       num_workers=self.read_threads,
                                       shuffle=False, collate_fn=collate_fn, drop_last=True)

    def __iter__(self):
        return torchiter(self.torchloader)

    def next(self):
        """
        :return:
            mx.io.DataBatch(data = [data], label = [label]) 
                detailed explanation later        
        :raises:
            StopIteration: 
                if data loader reaches epoch end
        """
        pass

    def __getitem__(self, index):
        """
        :param index(int): Index
        :return: 
            tuple: (data, label) where data and label are collections 
        """
        return self.dataset[index]

    @staticmethod
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
        # TODO: implement shuffle later
        # if self.random_shuffle:
        #     random.shuffle(self.loader_list)

    def __len__(self):
        return len(self.dataset)
'''