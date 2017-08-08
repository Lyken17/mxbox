#!/usr/bin/python
 # -*- coding: utf-8 -*-
import os.path as osp

import mxnet as mx
import mxbox
from mxbox import transforms


if __name__ == "__main__":
    test_root = "../../Data/TestFolder/"

    trans = transforms.Compose([
        transforms.mx.ToNdArray()
    ])
    dst = mxbox.datasets.CIFAR10('../../Data/',transform=trans, download=True)

    feedin_shapes = {
        'batch_size': 2,
        'data': [mx.io.DataDesc(name='data', shape=(2, 3, 32, 32), layout='NCHW')],
        'label': [mx.io.DataDesc(name='softmax_label', shape=(2, ), layout='N')]
    }

    loader = mxbox.DataLoader(dst, feedin_shapes, threads=1)

    for each in loader:
        print(each)