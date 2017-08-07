#!/usr/bin/python
 # -*- coding: utf-8 -*-
import os.path as osp

import mxnet as mx
import mxbox
from mxbox import transforms
from mxbox import models


if __name__ == "__main__":
    test_root = "../../Data/TestFolder/"

    trans = transforms.Compose([
        transforms.mx.ToNdArray()

    ])
    train = mxbox.datasets.CIFAR10('../../Data/', train=True, transform=trans, download=True)
    # valid = mxbox.datasets.CIFAR10('../../Data/', train=False, transform=trans, download=True)

    feedin_shapes = {
        'batch_size': 2,
        'data': [mx.io.DataDesc(name='data', shape=(2, 3, 32, 32), layout='NCHW')],
        'label': [mx.io.DataDesc(name='softmax_label', shape=(2, ), layout='N')]
    }

    train_loader = mxbox.DataLoader(train, feedin_shapes, threads=1)
    # valid_loader = mxbox.DataLoader(valid, feedin_shapes, threads=1)

    sym = models.resnet_getsym(num_classes=10, num_layers=20, image_shape='3,28,28')

    devs = mx.cpu()

    model = mx.mod.Module(
        context=devs,
        symbol=sym
    )

    optimizer_params = {
        'learning_rate': 0.1,
        'momentum': 0.9,
        'wd': 1e-6,
        'multi_precision': True
    }
    
    model.fit(train_loader,
              begin_epoch=0,
              num_epoch=10,
              eval_metric=['accuracy'],
              optimizer = 'sgd',
              optimizer_params=optimizer_params,
    )